import os, json, asyncio, re, sys, logging
from collections.abc import Mapping, Sequence
import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# ----------------- Env setting -----------------
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# ----------------- Providers setting -----------------
# OpenAI
from openai import OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_DEFAULT = os.getenv("OPENAI_MODEL", "gpt-5-mini")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Gemini
import google.generativeai as genai
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL_DEFAULT = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL_DEFAULT = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3.1:free")
openrouter_client = None
if OPENROUTER_API_KEY:
    openrouter_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

# ----------------- MCP -----------------
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ----------------- Utils -----------------
def mcp_tool_to_openai(tool) -> dict:
    """
    Convert MCP tool metadata to an OpenAI-style function tool item.
    """
    params = (
        getattr(tool, "input_schema", None)
        or getattr(tool, "inputSchema", None)
        or {"type": "object", "properties": {}}
    )
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": getattr(tool, "description", "") or "",
            "parameters": params,
        },
    }

def extract_json_from_result(result: Any) -> Optional[dict]:
    """
    Extract {"structuredContent": {...}} → {...}
    """
    top = result
    if not isinstance(top, dict):
        try:
            top = json.loads(json.dumps(result, default=lambda o: getattr(o, "__dict__", str(o))))
        except Exception:
            top = {}
    sc = top.get("structuredContent")
    if isinstance(sc, dict):
        return sc
    return None

def tool_result_to_text(result: Any) -> str:
    content = getattr(result, "content", None)
    if not content and isinstance(result, dict):
        content = result.get("content")

    parts: List[str] = []
    if isinstance(content, list):
        for item in content:
            t = getattr(item, "type", None) or (isinstance(item, dict) and item.get("type"))
            if t == "text":
                parts.append(getattr(item, "text", None) or item.get("text", ""))
            elif t == "json":
                payload = getattr(item, "json", None) or item.get("json")
                parts.append(json.dumps(payload, ensure_ascii=False))
            else:
                parts.append(str(item))
    if parts:
        return "\n".join(p for p in parts if p is not None)

    try:
        return json.dumps(result, ensure_ascii=False)
    except Exception:
        return str(result)

def to_plain(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if dataclasses.is_dataclass(obj):
        return to_plain(dataclasses.asdict(obj))

    if isinstance(obj, Mapping):
        return {str(k): to_plain(v) for k, v in obj.items()}

    if isinstance(obj, Sequence) and not isinstance(obj, (bytes, bytearray, str)):
        return [to_plain(x) for x in obj]

    for attr in ("to_dict", "as_dict", "model_dump", "dict"):
        if hasattr(obj, attr) and callable(getattr(obj, attr)):
            try:
                return to_plain(getattr(obj, attr)())
            except Exception:
                pass

    if hasattr(obj, "__dict__"):
        try:
            return to_plain(vars(obj))
        except Exception:
            pass

    return str(obj)


_GEMINI_SCHEMA_KEEP = {"type", "description", "properties", "required", "items", "enum"}

def _sanitize_schema_for_gemini(node):
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k not in _GEMINI_SCHEMA_KEEP:
                continue
            if k in {"properties"} and isinstance(v, dict):
                out[k] = {pk: _sanitize_schema_for_gemini(pv) for pk, pv in v.items()}
            elif k in {"items"}:
                out[k] = _sanitize_schema_for_gemini(v)
            elif k in {"required"} and isinstance(v, list):
                out[k] = [str(x) for x in v if isinstance(x, (str, int, float))]
            elif k in {"enum"} and isinstance(v, list):
                out[k] = [x for x in v if isinstance(x, (str, int, float))]
            else:
                out[k] = v
        if out.get("type") == "object" and "properties" not in out:
            out["properties"] = {}
        return out
    elif isinstance(node, list):
        return [_sanitize_schema_for_gemini(x) for x in node]
    else:
        return node

SYSTEM_HINT = "\n".join([
    "TOOL USAGE RULES:",
    "- ALWAYS call a tool when the query involves counts, aggregations, dates, languages, places, or terms.",
    "- NEVER ask clarifying questions if the query already provides enough information for a best-guess.",
    "- Map keywords to parameters deterministically: twitter => collection=twitter; mastodon => collection=mastodon.",
    "- If the user asks for a trend over a year without granularity, default to aggregation_level=month.",
    "- If the collection is not explicitly given, default to collection='twitter' unless the user explicitly asks for multiple platforms (e.g., “compare platforms”).",
    "- Available collections (platforms): Twitter, Mastodon, Reddit, YouTube, BlueSky, Flickr.",
    "- When applying defaults, you must disclose the exact default values used and suggest alternatives the user can specify for further query",
    "- You may call tools multiple times in sequence to construct a complete answer.",
    "AVAILABLE TOOLS AND ARGUMENTS:",
    "- aio_version(): Get API version.",
    "- collection_summary({collection}): Summary (count/date range) for a collection.",
    "- aggregate_by_time({collection, start_date, end_date, aggregation_level=day|month|year, sentiment?=false}).",
    "- aggregate_day({collection, start_date, end_date, sentiment?=false}).",
    "- aggregate_month({collection, start_date, end_date, sentiment?=false}).",
    "- aggregate_year({collection, start_date, end_date, sentiment?=false}).",
    "- aggregate_seasonality({collection, startDate, endDate, aggregationLevel=dayofweek|hourofday, sentiment?=false}).",
    "- aggregate_language({collection, startDate, endDate, aggregationLevel=language|country|state|gccsa|suburb, sentiment?=false}).",
    "- aggregate_place({collection, startDate, endDate, aggregationLevel=country|state|gccsa|suburb|language, sentiment?=false}).",
    "- aggregate_terms_all({collection, start_date, end_date}).",
    "- aggregate_terms_specific({collection, start_date, end_date, terms='\"a,b,c\"'}).",
    "- nlp_terms_available({collection, day=YYYY-MM-DD}).",
    "- nlp_term_similarity({collection, day=YYYY-MM-DD, term, topk?=25}).",
    "ADDITIONAL RULES:",
    "- Dates are inclusive and must be in ISO format (YYYY-MM-DD).",
    "- sentiment?=true adds 'sentiment' and 'sentimentcount' fields.",
    "- Terms are Porter-stemmed; ensure correct stemming when querying terms.",
    "- On tool errors, adjust arguments and retry up to 5 times automatically.",
    "- If a query requires multiple steps, invoke tools sequentially until all data is obtained—do not ask the user for permission.",
    "- When a query requests a combination such as keyword + geography or keyword + language, and no single tool supports it, use a two-step (or more steps) process: ",
    "- When a term contains spaces (e.g., 'A B C'), prefer using aggregate_by_time with extra_params.q and wrap the phrase in double quotes (e.g., q='\"A B C\"')",
    "Output format needs to follow SUMMARY_PROMPT_DEFAULT",
])

SUMMARY_PROMPT_DEFAULT = (
    "First, output an 'MCP_CALLS' section listing ALL MCP tool calls (including retries)."
    "For each call, show: index, tool name, sanitized arguments, outcome, rows returned (if applicable), and a one-line note."
    "Sanitize arguments: lowercase collection, redact secrets, and truncate long arrays/maps (show first 3 items then '…')."
    "If defaults were applied, state them inline with the call. If a call failed, include the HTTP status/code and concise reason.\n\n"
    "Then, output a 'SUMMARY' section that clearly and completely describes the results. Include: "
    "total document volume (with units), key trends over time, exact dates of peaks and dips, breakdowns by top languages and places if available, and any notable anomalies or data-quality caveats."
    "Be strictly factual and precise; avoid speculation or unsupported inferences. "
    "If any defaults were applied, restate them and suggest alternative values the user can specify. "
    "If a tool error occurred, report the error code/message and propose a corrected follow-up query."
)

# ----------------- Gemini helpers -----------------
def oai_messages_to_gemini_contents(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    contents = []
    for m in messages:
        role = m.get("role", "user")
        text = m.get("content", "")

        if role in ("system", "user"):
            if isinstance(text, (list, tuple)):
                text = "\n".join(str(x) for x in text)
            contents.append({"role": "user", "parts": [{"text": str(text)}]})

        elif role == "assistant":
            if isinstance(text, (list, tuple)):
                text = "\n".join(str(x) for x in text)
            contents.append({"role": "model", "parts": [{"text": str(text)}]})

        elif role == "tool":
            name = m.get("name") or "tool"
            try:
                resp_obj = json.loads(text) if isinstance(text, str) else text
                if not isinstance(resp_obj, dict):
                    resp_obj = {"text": str(text)}
            except Exception:
                resp_obj = {"text": str(text)}

            contents.append({
                "role": "user",
                "parts": [{
                    "function_response": {
                        "name": name,
                        "response": resp_obj
                    }
                }]
            })

        else:
            if isinstance(text, (list, tuple)):
                text = "\n".join(str(x) for x in text)
            contents.append({"role": "user", "parts": [{"text": str(text)}]})
    return contents


def oai_tools_to_gemini_tools(oa_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tools: List[Dict[str, Any]] = []
    for t in oa_tools or []:
        if t.get("type") != "function":
            continue
        fn = t.get("function", {})
        raw_params = fn.get("parameters", {"type": "object", "properties": {}})
        clean_params = _sanitize_schema_for_gemini(raw_params)
        tools.append({
            "function_declarations": [{
                "name": fn.get("name"),
                "description": fn.get("description", ""),
                "parameters": clean_params,
            }]
        })
    return tools

def gemini_pick_tool_calls(resp) -> List[Tuple[str, Dict[str, Any]]]:
    out = []
    try:
        cand = resp.candidates[0]
        for part in cand.content.parts:
            fc = getattr(part, "function_call", None)
            if fc:
                name = fc.name
                raw_args = fc.args
                args = to_plain(raw_args) if raw_args is not None else {}
                if not isinstance(args, dict):
                    args = {"_": args}
                out.append((name, args))
    except Exception:
        pass
    return out

# ---- sanitize OpenAPI/JSON-Schema for google-generativeai ----
_GEMINI_ALLOWED_SCHEMA_KEYS = {
    "type", "format", "description",
    "properties", "required", "enum", "items",
    "anyOf", "oneOf", "allOf", "nullable", "default",
    "additionalProperties"
}

# ----------------- Provider runs -----------------
async def run_with_openai(model: str, messages: List[Dict[str, Any]], oa_tools: List[Dict[str, Any]],
                          session: ClientSession, max_tool_rounds: int, json_only: bool):
    """
    Multi-turn loop with OpenAI function calling and MCP tool execution.
    Returns a result bundle: {"tool_call", "aio_response", "llm_response", "logs"}.
    """
    assert openai_client is not None, "OPENAI_API_KEY not set"
    bundle = _new_result_bundle()
    bundle["logs"]["provider"] = "openai"
    bundle["logs"]["model"] = model

    for r in range(max_tool_rounds + 1):
        bundle["logs"]["rounds"] = r
        resp = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            tools=oa_tools,
            tool_choice="auto",
        )
        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)

        if not tool_calls:
            if msg.content:
                bundle["llm_response"] = msg.content
            return bundle

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                } for tc in tool_calls
            ]
        })

        for tc in tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            bundle["tool_call"] = {"name": name, "args": args}
            try:
                result = await session.call_tool(name, arguments=args)
                raw = extract_json_from_result(result)
                if raw is not None:
                    bundle["aio_response"] = raw
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "name": name, "content": tool_result_to_text(result)}
                )
            except Exception as e:
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "name": name, "content": f"[TOOL_ERROR:{name}] {e}"}
                )
    return bundle


async def run_with_gemini(model: str, messages: List[Dict[str, Any]], oa_tools: List[Dict[str, Any]],
                          session: ClientSession, max_tool_rounds: int, json_only: bool):
    assert GEMINI_API_KEY, "GEMINI_API_KEY not set"

    bundle = _new_result_bundle()
    bundle["logs"]["provider"] = "gemini"
    bundle["logs"]["model"] = model

    gmodel = genai.GenerativeModel(
        model,
        tools=oai_tools_to_gemini_tools(oa_tools),
        tool_config={"function_calling_config": {"mode": "AUTO"}}
    )
    contents = oai_messages_to_gemini_contents(messages)

    for r in range(max_tool_rounds + 1):
        bundle["logs"]["rounds"] = r
        resp = gmodel.generate_content(contents, safety_settings=None)
        tool_calls = gemini_pick_tool_calls(resp)

        if not tool_calls:
            txt = resp.text or ""
            if txt:
                bundle["llm_response"] = txt
            return bundle

        for name, args in tool_calls:
            bundle["tool_call"] = {"name": name, "args": args}
            try:
                result = await session.call_tool(name, arguments=args)
                tool_text = tool_result_to_text(result)
                raw = extract_json_from_result(result)
                if raw is not None:
                    bundle["aio_response"] = raw

                response_obj = raw
                if response_obj is None:
                    try:
                        parsed = json.loads(tool_text)
                        response_obj = parsed if isinstance(parsed, dict) else {"text": tool_text}
                    except Exception:
                        response_obj = {"text": tool_text}

                contents.append({
                    "role": "user",
                    "parts": [{
                        "function_response": {
                            "name": name,
                            "response": response_obj
                        }
                    }]
                })
            except Exception as e:
                err = f"[TOOL_ERROR:{name}] {e}"
                messages.append({"role": "tool", "tool_call_id": name, "name": name, "content": err})
                contents.append({
                    "role": "user",
                    "parts": [{"function_response": {"name": name, "response": {"error": err}}}]
                })
                continue
    return bundle


async def run_with_openrouter(model: str, messages: List[Dict[str, Any]], oa_tools: List[Dict[str, Any]],
                              session: ClientSession, max_tool_rounds: int, json_only: bool):
    assert openrouter_client is not None, "OPENROUTER_API_KEY not set"
    bundle = _new_result_bundle()
    bundle["logs"]["provider"] = "openrouter"
    bundle["logs"]["model"] = model

    for r in range(max_tool_rounds + 1):
        bundle["logs"]["rounds"] = r
        resp = openrouter_client.chat.completions.create(
            model=model,
            messages=messages,
            tools=oa_tools,
            tool_choice="auto",
            extra_headers={},
            extra_body={},
        )
        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)

        if not tool_calls:
            if msg.content:
                bundle["llm_response"] = msg.content
            return bundle

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                } for tc in tool_calls
            ]
        })

        for tc in tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            bundle["tool_call"] = {"name": name, "args": args}
            try:
                result = await session.call_tool(name, arguments=args)
                raw = extract_json_from_result(result)
                if raw is not None:
                    bundle["aio_response"] = raw
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "name": name, "content": tool_result_to_text(result)}
                )
            except Exception as e:
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "name": name, "content": f"[TOOL_ERROR:{name}] {e}"}
                )
    return bundle

# ----------------- Main chat_once -----------------
async def chat_once(
    user_query: str,
    server_dir: Path,
    server_script: str | None = "server.py",
    server_module: str | None = None,
    max_tool_rounds: int = 10,
    json_only: bool = True,
    provider: str = "gemini",
    model_override: Optional[str] = None
):
    script_path = (server_dir / (server_script or "server.py")).resolve()
    params = StdioServerParameters(
        command="uv",
        args=["--directory", str(server_dir), "run", str(script_path)],
        env=os.environ.copy(),
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_resp = await session.list_tools()
            tools = tools_resp.tools
            oa_tools = [mcp_tool_to_openai(t) for t in tools]

            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": SYSTEM_HINT},
                {"role": "user", "content": user_query},
            ]

            provider = (provider or "gemini").lower().strip()
            if provider == "openai":
                model = model_override or OPENAI_MODEL_DEFAULT
                bundle = await run_with_openai(model, messages, oa_tools, session, max_tool_rounds, json_only)
            elif provider == "gemini":
                model = model_override or GEMINI_MODEL_DEFAULT
                bundle = await run_with_gemini(model, messages, oa_tools, session, max_tool_rounds, json_only)
            elif provider == "openrouter":
                model = model_override or OPENROUTER_MODEL_DEFAULT
                bundle = await run_with_openrouter(model, messages, oa_tools, session, max_tool_rounds, json_only)
            else:
                raise ValueError("provider must be 'openai' or 'gemini' or 'openrouter'")

            return bundle

# ---- Result bundle helper ----
def _new_result_bundle():
    return {
        "tool_call": None,
        "aio_response": None,
        "llm_response": None,
        "logs": {
            "provider": None,
            "model": None,
            "rounds": 0
        }
    }

# ----------------- CLI -----------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--query", "-q", default="summary collection=twitter")
    p.add_argument("--provider", "-p", default="gemini", choices=["openai", "gemini", "openrouter"], help="LLM provider")
    p.add_argument("--model", "-m", default=None, help="override model name")
    p.add_argument("--server-dir", default=str((ROOT / "mcp_server").resolve()))
    p.add_argument("--server-script", default="server.py")
    p.add_argument("--server-module", default=None, help="ex: mcp_server.server")
    p.add_argument("--json-only", action=argparse.BooleanOptionalAction, default=True,
                   help="Print only JSON (default). Use --no-json-only to allow extra natural-language output.")
    p.add_argument("--max-tool-rounds", type=int, default=10)
    args = p.parse_args()

    result = asyncio.run(
        chat_once(
            user_query=args.query,
            server_dir=Path(args.server_dir),
            server_script=args.server_script if not args.server_module else None,
            server_module=args.server_module,
            json_only=args.json_only,
            max_tool_rounds=args.max_tool_rounds,
            provider=args.provider,
            model_override=args.model,
        )
    )
    payload = result.get("aio_response") or result
    if args.json_only:
        print(json.dumps(to_plain(payload), ensure_ascii=False))
    else:
        if result.get("llm_response"):
            print(result["llm_response"])
        print(json.dumps(to_plain({
            "tool_call": result.get("tool_call"),
            "aio_response": result.get("aio_response")
        }), ensure_ascii=False))