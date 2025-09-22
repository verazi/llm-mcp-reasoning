import os, json, asyncio, re, sys, logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ----------------- setup -----------------
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

logging.basicConfig(level=logging.INFO, stream=sys.stderr)

# ----------------- utils -----------------
def mcp_tool_to_openai(tool) -> dict:
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

# ----------------- extract json from call_tool return -----------------
def extract_json_from_result(result: Any) -> Optional[dict]:
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

# ---- stringify result for tool message fallback ----
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

# ---- connect to MCP server and run once ----
async def chat_once(user_query: str, server_dir: Path, server_script: str | None = "server.py",
                    server_module: str | None = None, max_tool_rounds: int = 3, json_only: bool = True,):

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
            print([t.name for t in tools])  # debugging
            oa_tools = [mcp_tool_to_openai(t) for t in tools]

            messages: List[Dict[str, Any]] = [
                {
                    "role": "system",
                    "content": (
                        "You can call tools via MCP. "
                        "Use `aio_version` to get API version. "
                        "Use `collection_summary` with {collection}. "
                        "Use `aggregate_by_time` with {collection, start_date, end_date, aggregation_level, sentiment}. "
                        "Use `aggregate_seasonality` with {collection, startDate, endDate, aggregationLevel=dayofweek|hourofday, sentiment}. "
                        "Or use convenience tools: `aggregate_day`, `aggregate_month`, `aggregate_year`. "
                        "If a tool returns an error, fix the arguments and retry up to 3 times."
                    ),
                },
                {"role": "user", "content": user_query},
            ]

            # multi-turn loop
            for _ in range(max_tool_rounds + 1):
                resp = openai.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=messages,
                    tools=oa_tools,
                    tool_choice="auto",
                )
                msg = resp.choices[0].message
                tool_calls = getattr(msg, "tool_calls", None)

                # no tool
                if not tool_calls:
                    print(msg.content or "")
                    break

                messages.append({"role": "assistant",
                                 "content": msg.content or "",
                                 "tool_calls": [
                                     {
                                         "id": tc.id,
                                         "type": "function",
                                         "function":{
                                             "name": tc.function.name,
                                             "arguments": tc.function.arguments,
                                         }
                                     }
                                     for tc in tool_calls
                                 ]
                                 })

                for tc in tool_calls:
                    name = tc.function.name
                    args = json.loads(tc.function.arguments or "{}")
                    try:
                        result = await session.call_tool(name, arguments=args)
                        raw = extract_json_from_result(result)
                        if raw is not None:
                            print(json.dumps(raw, ensure_ascii=False))
                            if json_only:
                                return
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "name": name,
                                "content": tool_result_to_text(result),
                            }
                        )
                    except Exception as e:
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "name": name,
                                "content": f"[TOOL_ERROR:{name}] {e}",
                            }
                        )

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--query", "-q", default="summary collection=twitter")
    p.add_argument("--server-dir", default=str((ROOT / "mcp_server").resolve()))
    p.add_argument("--server-script", default="server.py")
    p.add_argument("--server-module", default=None, help="ex: mcp_server.server")
    p.add_argument("--json-only", action=argparse.BooleanOptionalAction, default=True,
                   help="Print only JSON (default). Use --no-json-only to allow extra natural-language output.")
    args = p.parse_args()

    asyncio.run(
        chat_once(
            user_query=args.query,
            server_dir=Path(args.server_dir),
            server_script=args.server_script if not args.server_module else None,
            server_module=args.server_module,
            json_only=args.json_only,
        )
    )
