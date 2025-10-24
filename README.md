## Research Goal

Compare the reasoning performance of different LLMs under the **MCP** pipeline.  


## Project Structure
```bash
llm-mcp-reasoning/
│
├── mcp_client/                  # MCP client package
│   └── client.py                # Unified client MCP servers and LLMs
│
├── mcp_server/                  # MCP server package
│   ├── aio_client.py            # AIO API implementation
│   └── server.py                # Defines MCP tool endpoints
│
├── ui.py                       # Gradio-based voice interface for querying MCP tools
│
├── demo-final.ipynb            # Experiments Result
│
└── README.md

```



## Setup & Launch

### 1. Clone the repository

```bash
git clone https://github.com/verazi/llm-mcp-reasoning.git
cd llm-mcp-reasoning
```

Add the `AIO_API_KEY` key and `LLM_API_KEY` in env.

### 2. Create and activate a virtual environment
```bash
uv venv .venv --clear
source .venv/bin/activate
```

### 3. Install and upgrade dependencies

```bash
python -m ensurepip --upgrade 
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


## Interface


