import re, sys, json, asyncio, subprocess
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import gradio as gr
from faster_whisper import WhisperModel

# =========================
# Paths & constants
# =========================
PROJECT_ROOT = Path("/Users/tzuyu/PycharmProjects/llm-mcp-reasoning").resolve()
MCP_CLIENT_DIR = PROJECT_ROOT / "mcp_client"
MCP_SERVER_DIR = PROJECT_ROOT / "mcp_server"
CLIENT_PY = MCP_CLIENT_DIR / "client.py"
SERVER_SCRIPT = "server.py"
ASR_MODEL_SIZE = "small"
CSS_FONT_STACK = """
:root, .gradio-container, .gradio-container * {
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial,
               "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol" !important;
}
"""

# --- Helpers ---
def log_provider_model(provider: str, model: str) -> None:
    print(f"[Provider={provider}] [Model={model}]", file=sys.stderr)

def _status_text(provider: str, model: str) -> str:
    return f"**Status** · Provider=`{provider}` · Model=`{model}`"

def _metrics_text(provider: str, model: str, used_vad: bool, elapsed_s: float, note: str = "") -> str:
    note = f"\n- Note: {note}" if note else ""
    return (
        "### Run Metrics\n"
        f"- Provider: `{provider}`\n"
        f"- Model: `{model}`\n"
        f"- VAD: `{'ON' if used_vad else 'OFF'}`\n"
        f"- Elapsed: **{elapsed_s:.2f}s**"
        f"{note}"
    )

def _examples_block():
    return [
        "Show me the daily post counts on Twitter for March 2022.",
        "Summarize the trend of posts mentioning Formula during 2022.",
        "From January to June 2022 on Twitter, find the month with the highest activity and show the language breakdown for that month.",
        "List the top 20 terms used on Twitter between January and June 2022.",
    ]

# =========================
# ASR: speech to text
# =========================
_asr_model: Optional[WhisperModel] = None

def init_asr() -> WhisperModel:
    global _asr_model
    if _asr_model is None:
        _asr_model = WhisperModel(ASR_MODEL_SIZE, device="cpu", compute_type="int8")
    return _asr_model

def transcribe(audio_path: str, language: Optional[str], vad: bool) -> str:
    if not audio_path:
        return ""
    model = init_asr()
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        vad_filter=vad,
        language=None if (language == "auto" or not language) else language
    )
    text = "".join(seg.text for seg in segments).strip()
    return text


# =========================
# Try importing mcp_client
# =========================
def try_import_client():
    """
    Try to import a callable from mcp_client/client.py.
    If you have a function like `chat_once(...)`, we'll use it.
    Otherwise returns None and we'll fall back to subprocess.
    """
    try:
        import sys
        if str(MCP_CLIENT_DIR) not in sys.path:
            sys.path.insert(0, str(MCP_CLIENT_DIR))
        import client as mcp_cli  # noqa
        for name in ("chat_once", "run_once", "invoke"):
            if hasattr(mcp_cli, name):
                return getattr(mcp_cli, name)
        return None
    except Exception:
        return None

_client_entry = try_import_client()

def extract_all_json_from_text(s: str) -> list[dict]:
    found = []
    opens = [m.start() for m in re.finditer(r"\{", s)]
    for start in opens:
        depth = 0
        for i, ch in enumerate(s[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = s[start:i+1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            found.append(obj)
                    except Exception:
                        pass
                    break
    return found

def pick_primary_json(objs: list[dict]) -> dict:
    for o in objs:
        if isinstance(o, dict) and "aio_responses" in o:
            return o
    for o in objs:
        if isinstance(o, dict) and ("aio_response" in o or "tool_call" in o):
            return o
    return objs[-1] if objs else {}


MODEL_OPTIONS = {
    "gemini": ["gemini-2.5-flash", "gemini-2.0-flash"],
    "openrouter": ["deepseek/deepseek-chat-v3.1:free", "x-ai/grok-4-fast"]
}

# MODEL_OPTIONS = {
#     "openai": ["gpt-5-mini", "gpt-4o-mini", "gpt-4.1-mini"],
#     "gemini": ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash", "gemini-2.0-flash-lite"],
#     "openrouter": ["deepseek/deepseek-chat-v3.1:free", "x-ai/grok-4-fast"]
# }

# =========================
# Call MCP client
# =========================
def call_mcp_via_import(nl_query: str, want_natural_answer: bool, provider: str, model: str) -> Dict[str, Any]:
    log_provider_model(provider, model)

    if _client_entry is None:
        raise RuntimeError("No callable found in mcp_client; falling back to subprocess.")

    result = asyncio.run(_client_entry(
        user_query=nl_query,
        server_dir=MCP_SERVER_DIR,
        server_script=SERVER_SCRIPT,
        server_module=None,
        json_only=(not want_natural_answer),
        max_tool_rounds=5,
        provider=provider,
        model_override=model
    ))
    return {
        "tool_call": result.get("tool_call"),
        "aio_response": result.get("aio_response"),
        "aio_responses": result.get("aio_responses"),
        "llm_response": result.get("llm_response")
    }


def call_mcp_via_subprocess(nl_query: str, want_natural_answer: bool, provider: str, model: str) -> Dict[str, Any]:
    """
    Run your existing CLI:
      uv run --active python client.py --server-dir ../mcp_server --server-script server.py --query "<text>" [--no-json-only]
    Capture stdout, extract JSON, and return natural text if requested.
    """
    cmd = [
        "uv", "run", "--active", "python", "client.py",
        "--server-dir", "../mcp_server",
        "--server-script", SERVER_SCRIPT,
        "--query", nl_query,
        "--provider", provider,
        "--model", model
    ]
    if want_natural_answer:
        cmd.append("--no-json-only")

    proc = subprocess.run(
        cmd,
        cwd=str(MCP_CLIENT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    raw = proc.stdout or ""
    objs = extract_all_json_from_text(raw)
    obj = pick_primary_json(objs)

    natural = None
    if want_natural_answer:
        for k in ("llm_response", "final_answer", "message", "text"):
            if isinstance(obj, dict) and k in obj and isinstance(obj[k], str):
                natural = obj[k]
                break
        if natural is None:
            try:
                natural = raw
                for o in objs:
                    jtxt = json.dumps(o, ensure_ascii=False)
                    natural = natural.replace(jtxt, "")
                natural = natural.strip()
            except Exception:
                natural = raw.strip()

    return {
        "tool_call": obj.get("tool_call") if isinstance(obj, dict) else None,
        "aio_response": obj.get("aio_response") if isinstance(obj, dict) else None,
        "aio_responses": obj.get("aio_responses") if isinstance(obj, dict) else None,
        "llm_response": natural
    }


def call_mcp(nl_query: str, want_natural_answer: bool, provider: str, model: str) -> Dict[str, Any]:
    """
    Prefer import; fallback to subprocess that mirrors your current CLI usage.
    """
    if _client_entry:
        try:
            return call_mcp_via_import(nl_query, want_natural_answer, provider, model)
        except Exception:
            pass
    return call_mcp_via_subprocess(nl_query, want_natural_answer, provider, model)


# =========================
# Main pipeline: audio/text → MCP → UI outputs
# =========================
def run_pipeline(audio_path: Optional[str], nl_text: str, lang: str, use_vad: bool, want_natural_answer: bool, provider: str, model: str
                 ) -> Tuple[str, dict, str]:
    """
    1) Prefer user text; otherwise transcribe audio.
    2) Call MCP client with the text.
    3) Return: transcript, JSON payload, natural answer.
    """
    try:
        text = (nl_text or "").strip()
        if not text and audio_path:
            text = transcribe(audio_path, language=lang, vad=use_vad)
        if not text:
            return "", {"error": "No usable text. Please record audio or type a query."}, ""

        mcp_result = call_mcp(text, want_natural_answer, provider, model)

        if isinstance(mcp_result.get("aio_responses"), list) and mcp_result["aio_responses"]:
            display_json = mcp_result["aio_responses"]
        else:
            aio = mcp_result.get("aio_response")
            if isinstance(aio, dict) and "result" in aio:
                display_json = aio["result"]
            elif aio is not None:
                display_json = aio
            else:
                display_json = {"error": "No 'aio_responses' or 'aio_response' found."}

        # aio = mcp_result.get("aio_response")
        # if isinstance(aio, dict) and "result" in aio:
        #     display_json = aio["result"]
        # else:
        #     display_json = aio if aio is not None else {"error": "No 'result' field in aio_response."}

        llm_msg = (mcp_result.get("llm_response") or "").strip()
        return text, display_json, llm_msg

    except Exception as e:
        return "", {"error": str(e)}, ""

def launch():
    with gr.Blocks(title="MCP Voice UI", css=CSS_FONT_STACK, theme=gr.themes.Soft()) as demo:
        # --- Defaults ---
        default_provider = "gemini"
        default_model = MODEL_OPTIONS[default_provider][0]

        # --- Header / Status bar ---
        gr.Markdown("## MCP Voice UI")
        status_md = gr.Markdown(_status_text(default_provider, default_model))

        with gr.Row():
            # ===== Left column: Input =====
            with gr.Column(scale=5, min_width=420):
                with gr.Tab("Microphone"):
                    audio = gr.Audio(
                        sources=["microphone"], type="filepath",
                        label="Record (click to start/stop)"
                    )
                with gr.Tab("Text"):
                    text_in = gr.Textbox(
                        label="Or type your query here:",
                        lines=3,
                        placeholder="e.g., Show me the daily post counts on Twitter for March 2022."
                    )
                    gr.Examples(
                        examples=_examples_block(),
                        inputs=[text_in],
                        label="Quick examples"
                    )

                with gr.Accordion("Advanced settings", open=False):
                    with gr.Row():
                        lang = gr.Dropdown(
                            choices=["auto", "en"], value="auto", label="ASR Language"
                        )
                        use_vad = gr.Checkbox(value=True, label="Enable VAD")
                    with gr.Row():
                        provider = gr.Dropdown(
                            choices=list(MODEL_OPTIONS.keys()),
                            value=default_provider,
                            label="Provider"
                        )
                        model = gr.Dropdown(
                            choices=MODEL_OPTIONS[default_provider],
                            value=default_model,
                            label="Model"
                        )
                with gr.Row():
                    want_natural = gr.Checkbox(
                        value=True, label="Return natural language"
                    )
                with gr.Row():
                    btn = gr.Button("▶ Submit", variant="primary")
                    clear_btn = gr.Button("Clear")

            # ===== Right column: Outputs =====
            with gr.Column(scale=7, min_width=480):
                with gr.Row():
                    metrics_md = gr.Markdown("## Run Metrics")
                with gr.Row():
                    text_out = gr.Textbox(label="ASR Transcript", lines=2)
                with gr.Row():
                    llm_out = gr.Textbox(label="LLM Response", lines=7)
                with gr.Row():
                    json_out = gr.JSON(label="MCP Response")

        def on_provider_change(p: str):
            new_choices = MODEL_OPTIONS[p]
            new_value = new_choices[0]
            log_provider_model(p, new_value)
            return (
                gr.update(choices=new_choices, value=new_value),
                _status_text(p, new_value)
            )

        provider.change(
            fn=on_provider_change,
            inputs=provider,
            outputs=[model, status_md]
        )

        def on_model_change(p: str, m: str):
            log_provider_model(p, m)
            return _status_text(p, m)

        model.change(
            fn=on_model_change,
            inputs=[provider, model],
            outputs=status_md
        )

        import time
        def _wrapped_run_pipeline(audio_path, nl_text, lang_s, vad_on, want_natural_answer, p, m):
            start = time.time()
            log_provider_model(p, m)
            transcript, payload, natural = run_pipeline(
                audio_path, nl_text, lang_s, vad_on, want_natural_answer, p, m
            )
            elapsed = time.time() - start

            note = ""
            if isinstance(payload, dict) and "error" in payload:
                note = f"Error: {payload['error']}"
            metrics = _metrics_text(p, m, vad_on, elapsed, note=note)

            return transcript, payload, natural, metrics

        btn.click(
            fn=_wrapped_run_pipeline,
            inputs=[audio, text_in, lang, use_vad, want_natural, provider, model],
            outputs=[text_out, json_out, llm_out, metrics_md]
        )

        def _clear():
            return (
                None, "", "", {}, _status_text(default_provider, default_model),
                "### Run Metrics\n- Cleared."
            )

        clear_btn.click(
            fn=_clear,
            inputs=[],
            outputs=[audio, text_in, text_out, json_out, status_md, metrics_md]
        )

    log_provider_model(default_provider, default_model)
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)

if __name__ == "__main__":
    launch()
