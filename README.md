## Research Goal
Compare the reasoning performance of different LLMs under the **MCP** pipeline.  


## Project Structure


## Setup & Launch

### 1. Clone the repository

```bash
git clone https://github.com/verazi/llm-mcp-reasoning.git
cd llm-mcp-reasoning
```

### 2. Create and activate a virtual environment
```bash
uv venv .venv --clear
source .venv/bin/activate
```

### 3. Install and upgrade dependencies

```bash
python -m pip install -U pip wheel setuptools \
    "numpy>=2.0" "pandas>=2.2.2" "pyarrow>=17.0.0" \
    gradio "faster-whisper>=1.0.0" "huggingface_hub>=0.36.0"
```

This installs:
- Updated build tools (pip, wheel, setuptools)
- Core libraries: numpy, pandas, pyarrow
- UI framework: gradio
- Speech model: faster-whisper
- Model hub support: huggingface_hub

### 4. Run the UI
```bash
python ui.py
```
