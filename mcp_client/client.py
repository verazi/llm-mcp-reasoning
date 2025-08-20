import os, json, asyncio
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# ---- LLM setting ----
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---- 把 MCP tool 轉成 OpenAI 的 function schema ----
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

# ---- 把 MCP call_tool 回傳結果轉成文字（餵回 LLM）----
def tool_result_to_text(result: Any) -> str:
    # result.content 可能是 list[{"type":"text","text":"..."}, {"type":"json","json":{...}}]
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
    # 兜底：整包轉字串
    try:
        return json.dumps(result, ensure_ascii=False)
    except Exception:
        return str(result)

async def chat_once(
    user_query: str,
    server_dir: Path,
    server_script: str | None = "server.py",
    server_module: str | None = None,
    max_tool_rounds: int = 3,
):
    """用 stdio 啟 MCP server（透過 uv），把工具交給 LLM 決策，必要時呼叫工具多輪，回最終答案。"""
    # `uv run mcp dev <ABSOLUTE_PATH_TO_SERVER_PY>`
    script_path = (server_dir / (server_script or "server.py")).resolve()
    params = StdioServerParameters(
        command="uv",
        args=["--directory", str(server_dir), "run", str(script_path)],
        env=os.environ.copy(),
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 取得 tools，轉成 LLM 可用的 function schema
            tools_resp = await session.list_tools()
            tools = tools_resp.tools
            print([t.name for t in tools])
            oa_tools = [mcp_tool_to_openai(t) for t in tools]
            tools_by_name = {t.name: t for t in tools}

            messages: List[Dict[str, Any]] = [
                {
                    "role": "system",
                    "content": (
                        "You can call tools via MCP. "
                        "For AIO version use `aio_version`. "
                        "For collection summary use `collection_summary` with {collection}."
                        "If a tool returns an error, fix the arguments and retry up to 2 times."
                    ),
                },
                {"role": "user", "content": user_query},
            ]

            # 多輪工具迴圈
            for _ in range(max_tool_rounds + 1):
                resp = openai.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=messages,
                    tools=oa_tools,
                    tool_choice="auto",
                )
                msg = resp.choices[0].message
                tool_calls = getattr(msg, "tool_calls", None)

                # 沒有工具呼叫 → 視為最終答案
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

                # 逐一執行工具並把結果回餵
                for tc in tool_calls:
                    name = tc.function.name
                    args = json.loads(tc.function.arguments or "{}")
                    try:
                        result = await session.call_tool(name, arguments=args)
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
                                "content": f"ERROR calling tool {name}: {e}",
                            }
                        )
                # 迴圈會把工具結果再丟回去，讓模型決定是否還要下一輪

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--query", "-q", default="please give me the summary of twitter collection")
    p.add_argument("--server-dir", default=str((ROOT / "mcp-server").resolve()))
    # using script or module to start server
    p.add_argument("--server-script", default="server.py")
    p.add_argument("--server-module", default=None, help="ex: mcp_server.server")
    args = p.parse_args()

    asyncio.run(
        chat_once(
            user_query=args.query,
            server_dir=Path(args.server_dir),
            server_script=args.server_script if not args.server_module else None,
            server_module=args.server_module,
        )
    )



# OpenAI Testing
# import os
# import pathlib
#
# from dotenv import load_dotenv
# from openai import OpenAI
#
# ROOT = pathlib.Path(__file__).resolve().parent.parent
# load_dotenv(ROOT / ".env")
#
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#
# resp = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[{"role": "user", "content": "Hello from MCP client!"}]
# )
# print(resp.model)
# print(resp.usage)
# print(resp.choices[0].message)