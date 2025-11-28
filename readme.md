# Whisper STT → Clipboard

This project provides a GUI for real-time speech-to-text transcription using the Whisper model. The application listens to audio input, transcribes spoken words into text, and optionally pastes the transcribed text into the active window.

## Key Features

- **Real-time Transcription**: Live transcription of spoken words into text.
- **Model Selection**: Choose from different Whisper model sizes (tiny, base, small, medium, large-v3).
- **Partial Mode**: Transcribe speech in segments, useful for live preview.
- **Silence Detection**: Automatically detects silence to improve transcription accuracy.
- **Auto Paste**: Option to automatically paste transcribed text into the active window.
- **Remote Control**: Control the application via a socket interface for remote management.

## Why This Project Exists

This project aims to provide a user-friendly interface for real-time speech-to-text transcription, making it easier to transcribe spoken content into text. It is particularly useful for dictation, transcription, and other applications where real-time text generation is required. The application is designed to be lightweight and efficient, leveraging the power of the Whisper model for accurate transcription.
