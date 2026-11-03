# Depth Studio

A small local web UI for generating grayscale and colorized depth maps with
Depth Anything V2 Small.

## Requirements

- Python 3.10 or newer
- A working PyTorch installation (CPU or CUDA)

## Setup on Windows

Open PowerShell in this folder and install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If you want GPU acceleration, install the PyTorch build that matches your CUDA
setup by following the [official PyTorch instructions](https://pytorch.org/get-started/locally/).

## Run the UI

```powershell
python app.py
```

Open <http://127.0.0.1:8000> in your browser. The first depth generation
downloads the model from Hugging Face and may take a little longer. Keep the
terminal open while using the UI; press `Ctrl+C` there to stop the server.

You can also run inference from the command line:

```powershell
python infer.py path\to\image.jpg
```

The command-line script writes `depth_map_grayscale.png` and
`depth_map_colored.png` in the current folder. Those generated files and local
Python environments are excluded from Git.
