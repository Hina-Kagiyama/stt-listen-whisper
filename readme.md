# Overview

This project, named Whisper STT GUI, provides a graphical user interface (GUI) for controlling and interacting with the Whisper speech-to-text (STT) model. The application allows users to start, stop, and toggle the recording of audio and transcribes it into text using the Whisper model.

# Key Features

- **Graphical User Interface (GUI):** A user-friendly interface for controlling the Whisper STT model.
- **Model Selection:** Allows users to choose from different Whisper model variants (e.g., "tiny", "base", "small", "medium", "large-v3").
- **Partial Mode:** Live preview of transcriptions based on silence detection.
- **Auto-Paste:** Option to automatically paste recognized text into the active window (requires `xdotool` on X11).
- **Remote Control:** Ability to control the application via a socket interface.

# Motivation

The project aims to provide a convenient and user-friendly interface for using the Whisper speech-to-text model. It addresses the need for a more accessible and interactive way to transcribe audio into text, especially for those who prefer a graphical interface over command-line tools. The application also supports remote control, making it easier to integrate into larger workflows or scripts.

# Dependencies

- **Python 3:** The project is written in Python 3.
- **PyQt6:** For the graphical user interface.
- **torch:** For loading and using the Whisper model.
- **samplerate:** For audio resampling.
- **numpy:** For numerical operations.
- **subprocess:** For running system commands.
- **json:** For handling configuration files.
- **socket:** For inter-process communication (IPC).

# Usage

The Whisper STT GUI is intended to be used for transcribing audio into text using the Whisper model. Users can start, stop, and toggle the recording of audio through the GUI. The application also supports remote control via a socket interface, allowing for integration into larger workflows or scripts. Users can configure the model, silence sensitivity, and other settings through the GUI.
