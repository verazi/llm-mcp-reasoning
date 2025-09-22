import sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import logging
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base
from mcp_server.aio_client import AIOClient
from typing import Optional, Any, Literal

logging.basicConfig(level=logging.INFO, stream=sys.stderr)

mcp = FastMCP("AIO-API")

# ---------- helpers ----------
_client: Optional[AIOClient] = None
def get_client() -> AIOClient:
    global _client
    if _client is None:
        _client = AIOClient()
    return _client

# ------------------------------------------------------------------------
#                           Define Tools
# ------------------------------------------------------------------------

# Tool -- Authorization and Authentication
@mcp.tool()
def aio_version() -> str:
    """
    Retrieve the API version of the AIO service using:

        GET /version

    :return: A string containing the API version, e.g. "1.0.2-api"
    """
    client = get_client()
    return client._get("/version")

# Tool -- Summary of a collection
@mcp.tool()
def collection_summary(collection: str) -> dict[str, Any]:
    """
    The number of posts in a collection and the start and end dates of harvesting can be retrieved using:

        GET /analysis/aggregate/collections/{collection}/summary

    :param collection: Collection identifier (string)
    :return: A dictionary with keys: "startDate", "endDate", "count"
    """
    client = get_client()
    collection = collection.lower().strip()
    return client.collection_summary(collection)

# ------- Aggregation Tools -------
# Tool -- Aggregate by time
@mcp.tool()
def aggregate_by_time(collection: str, start_date: str, end_date: str,
                      aggregation_level: Literal["day", "month", "year"], sentiment: bool = False,
                      extra_params: dict[str, Any] | None = None, ) -> list[dict[str, Any]]:
    """
    Aggregate documents in a collection by time buckets.

    API:
        GET /analysis/aggregate/collections/{collection}/aggregation

    Parameters:
        collection (str): Collection identifier, e.g. "twitter".
        start_date (str): Start date (YYYY-MM-DD or ISO date-time), inclusive.
        end_date (str): End date (YYYY-MM-DD or ISO date-time), inclusive.
        aggregation_level (str): One of "day", "month", "year".
        sentiment (bool): If True, return sentiment stats instead of count.
        extra_params (dict): Additional query parameters (optional).

    Notes:
        - If aggregationLevel=month, the client automatically aligns to month start/end,
          and splits queries into chunks (to avoid overly large ranges).
        - If aggregationLevel=year, the client iterates by month and merges into annual totals.
        - When sentiment=True, the response includes "sentiment" and "sentimentcount".
          Otherwise, the response includes "count".

    Returns:
        list of dicts:
          - Count mode:     [{ "time": "YYYY-M-D", "count": <int> }, ...]
          - Sentiment mode: [{ "time": "YYYY-M-D", "sentiment": <float>, "sentimentcount": <int> }, ...]
    """
    client = get_client()
    collection = collection.lower().strip()
    level = str(aggregation_level).lower()
    if level not in {"day", "month", "year"}:
        raise ValueError("aggregation_level must be one of: day, month, year")

    return client.aggregate_by_time(collection=collection, start_date=start_date, end_date=end_date,
                                    aggregation_level=level, sentiment=bool(sentiment),
                                    extra_params=extra_params or None,)

@mcp.tool()
def aggregate_day(collection: str, start_date: str, end_date: str, sentiment: bool = False,
                  extra_params: dict[str, Any] | None = None,) -> list[dict[str, Any]]:
    """
    Wrapper for aggregate_by_time with aggregationLevel="day".
    """
    client = get_client()
    collection = collection.lower().strip()
    return client.aggregate_day(collection=collection, start_date=start_date, end_date=end_date,
                                sentiment=bool(sentiment), **(extra_params or {}),)

@mcp.tool()
def aggregate_month(collection: str, start_date: str, end_date: str, sentiment: bool = False,
                    extra_params: dict[str, Any] | None = None,) -> list[dict[str, Any]]:
    """
    Wrapper for aggregate_by_time with aggregationLevel="month".

    The client aligns to month start/end and handles chunked requests.
    """
    client = get_client()
    collection = collection.lower().strip()
    return client.aggregate_month(collection=collection, start_date=start_date, end_date=end_date,
                                  sentiment=bool(sentiment), **(extra_params or {}),)

# Tool -- Seasonality aggregation
@mcp.tool()
def aggregate_seasonality(collection: str, start_date: str, end_date: str,
                          aggregation_level: str,  # "dayofweek" | "hourofday"
                          sentiment: bool = False, extra_params: dict[str, Any] | None = None,) -> list[dict[str, Any]]:
    """
    Wrapper for client.aggregate_seasonality with aggregationLevel="dayofweek" or "hourofday".
    """
    client = get_client()
    collection = collection.lower().strip()
    return client.aggregate_seasonality(collection=collection, start_date=start_date, end_date=end_date,
                                        aggregation_level=aggregation_level, sentiment=bool(sentiment),
                                        **(extra_params or {}),)




# Tool -- Language aggregation
# Tool -- Place aggregation

# ------- Terms / Keywords -------
# Tool -- All terms (dictionary of stem → count)
# Tool -- Specific terms (daily series)

# ------- NLP / Embeddings / Topics -------
# Tool -- List terms available for a given day’s embedding model
# Tool -- Similar terms for a query word on a given day
# Tool -- Topic modelling (BERTopic)


# ------------------------------------------------------------------------
#                           Define Resources
# ------------------------------------------------------------------------

# Resource -- Collections list

# Resource Template -- Collection summary

# Resource Template -- Time aggregation

# Resource Template -- Seasonality

# Resource Template -- Language / Place aggregations

# Resource Template -- Terms

# Resource Template -- NLP / Topics


# ----------------------------------------------------------------
#                         Define Prompts
# ----------------------------------------------------------------

# -------- System guardrails --------
@mcp.prompt(title="System Guardrails")
def system_guardrails(
    json_only: bool = True,
    preferred_language: str = "English",
    timezone_hint: str = "Australia/Melbourne",
) -> list[base.Message]:
    """
    Returns a reusable System prompt with strict guardrails for tool usage,
    output formatting, validation, and privacy. Invoke this prompt and prepend
    the returned messages to your conversation to enforce consistent behavior.
    """
    rules = f"""You are a careful, precise assistant operating inside an MCP server.
    
            CORE BEHAVIOR
            - Be concise and unambiguous; prefer {preferred_language}.
            - If the user message is code- or data-focused, prioritize exactness over style.
            - Never invent API fields, endpoints, parameters, or data.

            TOOL USE
            - Call MCP tools only when needed and with minimal valid arguments.
            - Validate arguments before calling tools. Coerce simple types safely (e.g., "true"→true).
            - If a call fails, explain the error briefly and propose a corrected argument set.
            - Retry at most 3 times with improved arguments.
            
            OUTPUT FORMAT
            - Default: {"JSON only" if json_only else "Natural language allowed"}.
            - When returning tables, keep header names stable and consistent.
            - For code, provide a minimal runnable example with brief comments.
            - Do not leak access tokens, secrets, or internal file paths.

            DATE, TIME & TIMEZONE
            - Treat all relative dates using timezone: {timezone_hint}.
            - When clarifying dates, include the explicit YYYY-MM-DD form.
            - If results aggregate by hour, mention the timezone caveat once.
            
            DATA & ANALYTICS
            - For sentiment averages, mention the sample size caveat (sentimentcount) if present.
            - Do not over-interpret sparse buckets; call this out explicitly.

            ERROR HANDLING
            - Summarize root cause in ≤3 sentences.
            - Provide concrete fixes: exact keys, types, example values.
            - If uncertain, ask for the one most critical missing piece of info.

            PRIVACY & SAFETY
            - Do not include personally identifiable information unless explicitly provided by the user for output.
            - Never echo credentials or raw Authorization headers.

            STYLE
            - Use consistent key casing and parameter names that match the API.
            - Prefer numbered steps or bullet points for procedures.

            SUCCESS CRITERION
            - The user can copy-paste your output to run a request or include in a report with minimal edits.
            """
    msgs: list[base.Message] = [
        base.SystemMessage(rules.strip()),
    ]
    return msgs

if __name__ == "__main__":
    """Entry point for the direct execution server."""
    mcp.run()
    # uv run mcp dev mcp_server/server.py