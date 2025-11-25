# Whisper Transcription Project

This project uses the `faster-whisper` library to provide fast and accurate audio and video transcription. It can be used to transcribe single files or to watch a directory and automatically transcribe new files as they are added.

### Features

*   **Fast and Accurate Transcription:** Utilizes `faster-whisper` for efficient transcription.
*   **GPU Acceleration:** Supports NVIDIA GPUs (CUDA) for significantly faster performance.
*   **Automatic Device Detection:** Automatically detects and uses the best available device (CUDA, ROCm, CPU).
*   **File Watching:** Can monitor a directory and automatically transcribe new files.
*   **Markdown Output:** Saves transcriptions in a clean and readable Markdown format.

## Dependencies

### Software

*   **Python 3:** The project is written in Python.
*   **FFmpeg:** A cross-platform solution to record, convert and stream audio and video. It is used to extract audio from video files.
*   **CUDA and cuDNN (for NVIDIA GPUs):**
    *   **CUDA Toolkit:** A parallel computing platform and programming model developed by NVIDIA for general computing on graphical processing units (GPUs).
    *   **cuDNN (CUDA Deep Neural Network library):** A GPU-accelerated library of primitives for deep neural networks. It is required for `faster-whisper` to run on an NVIDIA GPU.

### Python Packages

The following Python packages are required:

*   `faster-whisper`: The core transcription library.
*   `numpy`: A fundamental package for scientific computing with Python.
*   `nvidia-cublas-cu12` and `nvidia-cudnn-cu12`: CUDA and cuDNN libraries for NVIDIA GPUs.
*   `watchdog`: A library for monitoring file system events.
*   `pyinstaller`: A tool to package Python applications into standalone executables.

## Setup and Installation

1.  **Create a Virtual Environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Install Dependencies:**

    *   **On Arch Linux:**

        The `requirement.sh` script can be used to install all the necessary dependencies:

        ```bash
        bash requirement.sh
        ```

    *   **On other Linux distributions and Windows:**

        *   Install **FFmpeg**, **CUDA**, and **cuDNN** by following the official instructions for your operating system.
        *   Install the required Python packages using `pip`:

            ```bash
            pip install --upgrade pip
            pip install faster-whisper numpy nvidia-cublas-cu12 nvidia-cudnn-cu12 watchdog pyinstaller
            ```

3.  **Configure `path.json`:**

    Create a file named `path.json` in the root directory of the project with the following content:

    ```json
    {
        "whisper_model": "whisperModels/faster-whisper-small",
        "save_path": "transcriptions",
        "WATCH_DIR": "watch",
        "TRANSCRIPT_SCRIPT": "src/transcript.py",
        "PYTHON_EXEC": "venv/bin/python"
    }
    ```

    *   `whisper_model`: The path to the `faster-whisper` model.
    *   `save_path`: The directory where the transcription files will be saved.
    *   `WATCH_DIR`: The directory to watch for new files.
    *   `TRANSCRIPT_SCRIPT`: The path to the `transcript.py` script.
    *   `PYTHON_EXEC`: The path to the Python executable in your virtual environment.

## Usage

### Transcribing a Single File

To transcribe a single audio or video file, run the `transcript.py` script with the file path as an argument:

```bash
python src/transcript.py /path/to/your/audio_or_video.mp3
```

The transcription will be saved in the directory specified in `save_path` in the `path.json` file.

### Watching a Directory

To automatically transcribe new files added to a directory, run the `watch.py` script:

```bash
python src/watch.py
```

The script will watch the directory specified in `WATCH_DIR` in the `path.json` file. When a new file is added, it will be automatically transcribed.

## Whisper Models

The `faster-whisper` library uses the same models as the original Whisper model. You can download the models from the [Hugging Face model repository](https://huggingface.co/guillaumekln).

The available models are:

*   `tiny`
*   `base`
*   `small`
*   `medium`
*   `large`

The larger the model, the more accurate the transcription, but it will also be slower and require more memory. The `small` model is a good balance between accuracy and performance.
