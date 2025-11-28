## Overview
This project provides a GUI for real-time speech-to-text (STT) transcription using the Whisper model, with the ability to control the transcription process via a socket interface. The application allows users to start, stop, and toggle the transcription process, and it supports various configuration options for the transcription behavior.

## Key Features
- Real-time speech-to-text transcription using the Whisper model.
- GUI for controlling the transcription process.
- Socket-based remote control for starting, stopping, and toggling transcription.
- Configurable model selection, partial mode, silence sensitivity, and extra command filters.
- Auto-paste functionality for copying recognized text to the system clipboard.

## Motivation
The project aims to provide a user-friendly interface for real-time speech-to-text transcription, enabling users to transcribe spoken content into text with minimal setup and configuration. The socket-based remote control allows for integration with other applications or scripts, making it versatile for various use cases.

## Dependencies
- Python 3
- PyQt6
- PyAudio
- Torch
- Whisper model (downloaded from the Hugging Face Model Hub)
- Sample rate conversion library (libsamplerate)

## Usage
The project is intended to be used as a speech-to-text transcription tool with a graphical user interface. Users can start, stop, and toggle the transcription process via the GUI or through a socket interface. The application supports various configuration options to customize the transcription behavior, such as selecting the Whisper model, enabling partial mode, and configuring silence sensitivity. Additionally, the auto-paste functionality can be used to copy recognized text to the system clipboard.
