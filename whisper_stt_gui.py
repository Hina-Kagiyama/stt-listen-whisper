#!/usr/bin/env python3
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QTextEdit,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QSlider,
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
import torch
import whisper
import samplerate
import numpy as np
import socket
import threading
import gc
import subprocess
import json
import os
import sys

ZH_SIMPLIFIED_PROMPT = (
    "请使用简体中文字符，在中文词汇附近使用中文标点，不要使用繁体字。"
)

# =========================
# IPC CONFIG
# =========================

SOCKET_PATH = "/tmp/whisper_stt_socket"

# =========================
# CONFIG CONSTANTS
# =========================

MODEL_CHOICES = ["tiny", "base", "small", "medium", "large-v3"]
WHISPER_MODEL_DEFAULT = "small"

# Language code (e.g. "zh" for Chinese, "en" for English, None = auto-detect)
WHISPER_LANGUAGE = "zh"

# Internal processing rate for Whisper input
TARGET_RATE = 16000

# arecord device: None = default ALSA device (usually PipeWire's "default")
ARECORD_DEVICE = None  # e.g. "pipewire" or "hw:0,0"

# Bytes per read from arecord: frames * channels * bytes_per_sample
DEFAULT_BLOCK_FRAMES = 2048

# Silence detection
MIN_SEGMENT_MS = 300             # minimum segment length to transcribe
BASE_MIN_THRESHOLD = 0.005       # absolute minimum RMS threshold
DEFAULT_SILENCE_SLIDER = 100     # 0–200 (%), maps to multiplier


# =========================
# Helper: detect default input sample rate
# =========================

def detect_default_sample_rate():
    """Try to detect default capture sample rate from arecord + PipeWire."""
    cmd = ["arecord", "-q", "--dump-hw-params"]
    if ARECORD_DEVICE is not None:
        cmd.extend(["-D", ARECORD_DEVICE])

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        out, _ = proc.communicate(timeout=1.0)

        # Scan for: RATE: min=xxxx max=xxxx
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("RATE:"):
                try:
                    parts = line.split()
                    min_rate = int(parts[1].split("=")[1])
                    max_rate = int(parts[2].split("=")[1])
                    if min_rate == max_rate:
                        return min_rate
                    return max_rate
                except Exception:
                    pass
    except Exception:
        pass

    return 44100  # fallback safe rate


ARECORD_RATE = detect_default_sample_rate()
FALLBACK_RATE_USED = (ARECORD_RATE == 44100)
BLOCK_BYTES = DEFAULT_BLOCK_FRAMES * 2  # mono * int16 (2 bytes)
print("Detected input sample rate:", ARECORD_RATE)


# =========================
# Helper: fp16 support
# =========================

def supports_fp16():
    """Return True if running with a CUDA GPU (fp16 suitable)."""
    return torch.cuda.is_available()


# =========================
# Whisper worker
# =========================

class WhisperWorker(QThread):
    partial_text = pyqtSignal(str)
    final_text = pyqtSignal(str)

    def __init__(
        self,
        whisper_model,
        partial_mode=False,
        silence_multiplier=1.7,
        silence_window_ms=200,
        parent=None,
    ):
        super().__init__(parent)
        self.whisper_model = whisper_model
        self.partial_mode = partial_mode
        self.silence_multiplier = silence_multiplier
        self.silence_window_ms = silence_window_ms
        self.min_segment_ms = MIN_SEGMENT_MS

        self._running = False
        self.proc = None
        self.fp16 = supports_fp16()

        # 16 kHz audio buffer
        self.audio_16k = None

        # Segmentation state
        self.segment_start_sample = 0  # start of current speech segment
        self.in_silence = True        # start as "in silence"
        self.partial_accumulated_text = ""

        # Dynamic noise level (RMS), learned from silence
        self.noise_rms = None

    # ----- Silence detection helpers -----

    def _tail_rms(self):
        """Compute RMS of the tail window of audio_16k."""
        if self.audio_16k is None:
            return None

        window_samples = int(self.silence_window_ms * TARGET_RATE / 1000)
        if window_samples <= 0 or self.audio_16k.shape[0] < window_samples:
            return None

        tail = self.audio_16k[-window_samples:]
        return float(np.sqrt(np.mean(tail ** 2)))

    def _is_tail_silence(self):
        """Check if the tail of audio_16k is 'silent' based on dynamic noise floor."""
        tail_rms = self._tail_rms()
        if tail_rms is None:
            return False

        # Initialize noise estimate with first RMS we see
        if self.noise_rms is None:
            self.noise_rms = tail_rms

        # Dynamic threshold
        dynamic_threshold = max(
            BASE_MIN_THRESHOLD,
            self.noise_rms * self.silence_multiplier
        )

        is_silence = tail_rms < dynamic_threshold

        # If we think it's silence, update noise floor slowly
        if is_silence:
            self.noise_rms = 0.9 * self.noise_rms + 0.1 * tail_rms

        return is_silence

    # ----- Transcription helpers -----

    def _transcribe_segment(self, start_sample, end_sample):
        segment = self.audio_16k[start_sample:end_sample].astype(np.float32)
        try:
            result = self.whisper_model.transcribe(
                segment,
                language=WHISPER_LANGUAGE,
                fp16=self.fp16,
                condition_on_previous_text=False,
                verbose=False,
                initial_prompt=ZH_SIMPLIFIED_PROMPT,
            )
            text = result.get("text", "").strip()
        except Exception as e:
            print("Whisper segment error:", e)
            text = ""
        return text

    def _maybe_finish_segment_on_silence(self):
        """
        If we've just transitioned from speech to silence and the segment is
        long enough, transcribe that segment and emit partial text.

        Important: we trim off the last silence window so we don't feed
        the explicit silence fragment into Whisper.
        """
        if self.audio_16k is None:
            return

        total_samples = self.audio_16k.shape[0]
        window_samples = int(self.silence_window_ms * TARGET_RATE / 1000)
        if window_samples <= 0:
            window_samples = 1

        # Trim off the last silence window from the segment we send to Whisper
        end_sample = max(self.segment_start_sample,
                         total_samples - window_samples)

        min_segment_samples = int(self.min_segment_ms * TARGET_RATE / 1000)
        segment_len = end_sample - self.segment_start_sample
        if segment_len < min_segment_samples:
            # Too short; ignore this, keep segment_start_sample as is
            return

        text = self._transcribe_segment(self.segment_start_sample, end_sample)
        if not text:
            return

        if self.partial_accumulated_text:
            self.partial_accumulated_text += "\n"
        self.partial_accumulated_text += text

        # Next segment starts after this silence (at current total_samples)
        self.segment_start_sample = total_samples

        self.partial_text.emit(self.partial_accumulated_text)

    # ----- Main thread loop -----

    def run(self):
        self._running = True

        # Build arecord command
        cmd = [
            "arecord",
            "-f", "S16_LE",           # 16-bit PCM
            "-c", "1",                # mono
            "-r", str(ARECORD_RATE),  # sample rate
            "-t", "raw",              # raw bytes to stdout
            "-q",                     # quiet (no arecord banner)
        ]
        if ARECORD_DEVICE is not None:
            cmd.extend(["-D", ARECORD_DEVICE])

        print("Starting arecord:", " ".join(cmd))
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            self.final_text.emit(
                "ERROR: 'arecord' not found. Install 'alsa-utils'.")
            return

        try:
            while self._running:
                if self.proc.poll() is not None:
                    # arecord has exited
                    break

                data = self.proc.stdout.read(BLOCK_BYTES)
                if not data:
                    # No more data
                    break

                # data is int16 PCM bytes -> float32 [-1, 1]
                int16_samples = np.frombuffer(
                    data, dtype=np.int16).astype(np.float32)
                float_samples = int16_samples / 32767.0

                # Resample to TARGET_RATE using libsamplerate
                ratio = TARGET_RATE / float(ARECORD_RATE)
                resampled = samplerate.resample(
                    float_samples, ratio, "sinc_best").astype(np.float32)

                # Append to 16 kHz buffer
                if self.audio_16k is None:
                    self.audio_16k = resampled
                    self.segment_start_sample = 0
                else:
                    self.audio_16k = np.concatenate(
                        [self.audio_16k, resampled])

                # Partial mode: silence-based segmentation with dynamic noise
                if self.partial_mode:
                    is_silence = self._is_tail_silence()

                    # State machine: detect speech -> silence transition
                    if not is_silence and self.in_silence:
                        # We just started speaking
                        self.in_silence = False

                    elif is_silence and not self.in_silence:
                        # We just transitioned from speech to silence
                        self.in_silence = True
                        self._maybe_finish_segment_on_silence()

        finally:
            # Ensure arecord is stopped
            if self.proc is not None and self.proc.poll() is None:
                try:
                    self.proc.terminate()
                except Exception:
                    pass
                try:
                    self.proc.wait(timeout=1.0)
                except Exception:
                    pass
                self.proc = None

            if self.audio_16k is None or self.audio_16k.shape[0] == 0:
                self.final_text.emit("")
                return

            total_samples = self.audio_16k.shape[0]
            print(
                f"Total audio length: {total_samples / TARGET_RATE:.2f} seconds")

            # ----- Final transcription behavior -----
            if not self.partial_mode:
                # Simple mode: one transcription over all audio
                try:
                    result = self.whisper_model.transcribe(
                        self.audio_16k.astype(np.float32),
                        language=WHISPER_LANGUAGE,
                        fp16=self.fp16,
                        verbose=False,
                        initial_prompt=ZH_SIMPLIFIED_PROMPT,
                    )
                    text = result.get("text", "").strip()
                except Exception as e:
                    text = f"ERROR running Whisper: {e}"
                self.final_text.emit(text)
                return

            # Partial mode: respect segmentation and avoid a full re-transcribe
            texts = []
            if self.partial_accumulated_text:
                texts.append(self.partial_accumulated_text)

            # Transcribe the last tail segment (if any), optionally trimming trailing silence
            min_segment_samples = int(self.min_segment_ms * TARGET_RATE / 1000)
            remaining_len = total_samples - self.segment_start_sample
            if remaining_len >= min_segment_samples:
                window_samples = int(
                    self.silence_window_ms * TARGET_RATE / 1000)
                if window_samples <= 0:
                    window_samples = 1

                # Decide whether the tail is silence; if so, trim it
                if self._is_tail_silence():
                    tail_end = max(self.segment_start_sample,
                                   total_samples - window_samples)
                else:
                    tail_end = total_samples

                if tail_end - self.segment_start_sample >= min_segment_samples:
                    tail_text = self._transcribe_segment(
                        self.segment_start_sample, tail_end
                    )
                    if tail_text:
                        texts.append(tail_text)

            final_text = "\n".join(texts).strip()
            self.final_text.emit(final_text)

    def stop(self):
        # Ask loop to exit
        self._running = False
        # Also stop arecord promptly
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass


# =========================
# GUI + CONFIG
# =========================

class MainWindow(QWidget):
    # Remote commands from IPC thread ("start", "stop", "toggle")
    remote_command = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = os.path.join(self.script_dir, "model")
        os.makedirs(self.model_dir, exist_ok=True)

        # Config path
        self.config_path = os.path.join(self.script_dir, "stt_config.json")
        self.config = {
            "model_name": WHISPER_MODEL_DEFAULT,
            "partial_mode": False,
            "silence_slider": DEFAULT_SILENCE_SLIDER,
            "auto_paste": False,
        }
        self.load_config()

        self.setWindowTitle("Whisper STT → Clipboard")
        self.whisper_model = None
        self.worker = None
        self.loading_model = False

        layout = QVBoxLayout(self)

        self.status_label = QLabel("Loading Whisper model, please wait…")
        layout.addWidget(self.status_label)

        # Show detected sample rate
        rate_text = f"Detected input sample rate: {ARECORD_RATE} Hz"
        if FALLBACK_RATE_USED:
            rate_text += " (fallback)"
        self.rate_label = QLabel(rate_text)
        layout.addWidget(self.rate_label)

        # Model selection row
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(MODEL_CHOICES)
        model_name = self.config.get("model_name", WHISPER_MODEL_DEFAULT)
        if model_name not in MODEL_CHOICES:
            model_name = WHISPER_MODEL_DEFAULT
        idx = self.model_combo.findText(model_name)
        if idx < 0:
            idx = 0
        self.model_combo.setCurrentIndex(idx)
        model_row.addWidget(self.model_combo)
        model_row.addStretch()
        layout.addLayout(model_row)

        self.button = QPushButton("Start recording")
        self.button.setEnabled(False)
        self.button.clicked.connect(self.on_button_clicked)
        layout.addWidget(self.button)

        # Partial mode checkbox
        self.partial_checkbox = QCheckBox(
            "Partial mode (silence-based live preview)"
        )
        self.partial_checkbox.setChecked(
            self.config.get("partial_mode", False)
        )
        self.partial_checkbox.stateChanged.connect(
            self.on_partial_mode_changed
        )
        layout.addWidget(self.partial_checkbox)

        # Auto paste checkbox
        self.paste_checkbox = QCheckBox(
            "Paste into active window after dictation (requires xdotool on X11)"
        )
        self.paste_checkbox.setChecked(self.config.get("auto_paste", False))
        self.paste_checkbox.stateChanged.connect(self.on_paste_mode_changed)
        layout.addWidget(self.paste_checkbox)

        # Silence sensitivity slider
        self.slider_label = QLabel()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(200)
        slider_val = int(self.config.get(
            "silence_slider", DEFAULT_SILENCE_SLIDER))
        slider_val = max(self.slider.minimum(), min(
            self.slider.maximum(), slider_val))
        self.slider.setValue(slider_val)
        self.slider.valueChanged.connect(self.on_slider_changed)
        self.update_slider_label(slider_val)
        layout.addWidget(self.slider_label)
        layout.addWidget(self.slider)

        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Recognized text will appear here.")
        layout.addWidget(self.text_edit)

        # IPC status label
        self.ipc_label = QLabel(
            f"Remote control socket: {SOCKET_PATH}"
        )
        layout.addWidget(self.ipc_label)

        self.resize(520, 560)

        # Connect combo AFTER UI is built to avoid double-loads
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)

        # Connect remote command signal
        self.remote_command.connect(self.on_remote_command)

        # Start IPC server
        self.start_ipc_server()

        # Load Whisper model
        self.load_model()

    # ----- IPC server -----

    def start_ipc_server(self):
        # Remove old socket if left over
        try:
            if os.path.exists(SOCKET_PATH):
                os.remove(SOCKET_PATH)
        except Exception as e:
            print("Failed to remove old socket:", e)

        def server_loop():
            try:
                srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                srv.bind(SOCKET_PATH)
                os.chmod(SOCKET_PATH, 0o600)  # user-only
                srv.listen(1)
            except Exception as e:
                print("IPC server setup error:", e)
                return

            while True:
                try:
                    conn, _ = srv.accept()
                except Exception as e:
                    print("IPC accept error:", e)
                    break
                try:
                    data = conn.recv(32)
                    cmd = data.decode(errors="ignore").strip().lower()
                    if cmd in ("start", "stop", "toggle"):
                        self.remote_command.emit(cmd)
                except Exception as e:
                    print("IPC recv error:", e)
                finally:
                    try:
                        conn.close()
                    except Exception:
                        pass

        t = threading.Thread(target=server_loop, daemon=True)
        t.start()

    def on_remote_command(self, cmd: str):
        # Called in GUI thread via signal
        if self.loading_model:
            # Ignore commands while loading model
            return

        if cmd == "toggle":
            if not self.worker or not self.worker.isRunning():
                self.start_recording()
            else:
                self.stop_recording()
        elif cmd == "start":
            if not self.worker or not self.worker.isRunning():
                self.start_recording()
        elif cmd == "stop":
            if self.worker and self.worker.isRunning():
                self.stop_recording()

    # ----- Config -----
    def load_config(self):
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.config.update(data)
        except Exception as e:
            print("Failed to load config:", e)

    def save_config(self):
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print("Failed to save config:", e)

    # ----- Model loading -----
    def load_model(self):
        self.loading_model = True
        model_name = self.config.get("model_name", WHISPER_MODEL_DEFAULT)
        self.button.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.slider.setEnabled(False)
        self.paste_checkbox.setEnabled(False)
        self.partial_checkbox.setEnabled(False)
        self.status_label.setText(
            f"Loading Whisper model '{model_name}' from ./model …"
        )
        QApplication.processEvents()

        try:
            # Explicitly delete old model to free RAM/VRAM
            if self.whisper_model is not None:
                del self.whisper_model
                self.whisper_model = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            self.whisper_model = whisper.load_model(
                model_name,
                download_root=self.model_dir,
            )
            self.status_label.setText(
                f"Ready. Model: {model_name}. Click 'Start recording' and speak."
            )
            self.button.setEnabled(True)
        except Exception as e:
            self.whisper_model = None
            self.status_label.setText(f"Error loading Whisper model: {e}")
            self.button.setEnabled(False)
        finally:
            self.loading_model = False
            self.model_combo.setEnabled(True)
            self.slider.setEnabled(True)
            self.paste_checkbox.setEnabled(True)
            self.partial_checkbox.setEnabled(True)

    # ----- UI handlers -----
    def on_partial_mode_changed(self, state):
        self.config["partial_mode"] = bool(state)
        self.save_config()

    def on_paste_mode_changed(self, state):
        self.config["auto_paste"] = bool(state)
        self.save_config()

    def on_model_changed(self, index):
        if self.loading_model:
            return  # ignore signal fired while we're loading

        new_model = self.model_combo.currentText()
        if self.worker and self.worker.isRunning():
            self.status_label.setText("Stop recording before changing model.")
            curr = self.config.get("model_name", WHISPER_MODEL_DEFAULT)
            idx = self.model_combo.findText(curr)
            if idx >= 0:
                self.model_combo.blockSignals(True)
                self.model_combo.setCurrentIndex(idx)
                self.model_combo.blockSignals(False)
            return

        self.config["model_name"] = new_model
        self.save_config()
        self.load_model()

    def update_slider_label(self, value: int):
        self.slider_label.setText(f"Silence sensitivity: {value}%")

    def on_slider_changed(self, value: int):
        self.update_slider_label(value)
        self.config["silence_slider"] = int(value)
        self.save_config()

    def on_button_clicked(self):
        if not self.worker or not self.worker.isRunning():
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        if not self.whisper_model or self.loading_model:
            return

        self.text_edit.clear()
        self.status_label.setText(
            "🎙 Recording… click 'Stop recording' when you’re done."
        )
        self.button.setText("Stop recording")

        partial_mode = self.partial_checkbox.isChecked()
        self.config["partial_mode"] = partial_mode

        slider_value = int(self.config.get(
            "silence_slider", DEFAULT_SILENCE_SLIDER))
        # Map 0–200 → multiplier ≈ 1.0–5.0
        silence_multiplier = 1.0 + slider_value * 0.02
        silence_window_ms = 200  # fixed window; could be another slider if you want

        self.save_config()

        self.worker = WhisperWorker(
            self.whisper_model,
            partial_mode=partial_mode,
            silence_multiplier=silence_multiplier,
            silence_window_ms=silence_window_ms,
        )
        self.worker.partial_text.connect(self.on_partial_text)
        self.worker.final_text.connect(self.on_final_text)
        self.worker.start()

    def stop_recording(self):
        if self.worker and self.worker.isRunning():
            self.status_label.setText("Stopping & transcribing, please wait…")
            self.button.setEnabled(False)
            self.worker.stop()

    def on_partial_text(self, text: str):
        self.text_edit.setPlainText(text)

    def trigger_paste(self):
        """
        Best-effort auto-paste using xdotool (X11).
        On Wayland this may do nothing, but it won't crash the app.
        """
        try:
            subprocess.Popen(["xdotool", "key", "ctrl+v"])
            self.status_label.setText("Done. Text copied & paste triggered.")
        except FileNotFoundError:
            self.status_label.setText(
                "Text copied. Auto-paste failed (xdotool not found)."
            )
        except Exception as e:
            self.status_label.setText(
                f"Text copied. Auto-paste error: {e}"
            )

    def on_final_text(self, text: str):
        self.button.setEnabled(True)
        self.button.setText("Start recording")

        if text:
            self.text_edit.setPlainText(text)
            if text.startswith("ERROR"):
                self.status_label.setText(text)
            else:
                try:
                    clipboard = QApplication.clipboard()
                    clipboard.setText(text)
                    self.status_label.setText(
                        "Done. Text copied to clipboard."
                    )
                    if self.config.get("auto_paste", False):
                        self.trigger_paste()
                except Exception as e:
                    self.status_label.setText(
                        f"Done, but clipboard failed: {e}"
                    )
        else:
            self.text_edit.setPlainText("(no speech recognized)")
            self.status_label.setText("No speech recognized. Try again.")

    def closeEvent(self, event):
        # Cleanly stop worker if closing while recording
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1000)

        # Remove socket file on exit
        try:
            if os.path.exists(SOCKET_PATH):
                os.remove(SOCKET_PATH)
        except Exception:
            pass

        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
