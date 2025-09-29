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
    Retrieve the API version of the AIO service.

    Endpoint
    --------
    GET /version

    Returns
    -------
    str
    API version string, e.g., "1.0.2-api".

    Examples
    --------
    - NL: "what API version is the AIO server?" -> call this tool directly.
    """
    client = get_client()
    return client._get("/version")

# Tool -- Summary of a collection
@mcp.tool()
def collection_summary(collection: str) -> dict[str, Any]:
    """
    Summary statistics for a collection: total documents and harvest window.

    Endpoint
    --------
    GET /analysis/aggregate/collections/{collection}/summary

    Parameters
    ----------
    collection : {"Twitter", "Mastodon", "Reddit", "YouTube", "BlueSky", "Flickr"}
    Collection identifier. The model MUST map the literal word "twitter" in the
    user query to collection="twitter" (same for other platforms).

    Returns
    -------
    dict
    {"startDate": "YYYY-MM-DD", "endDate": "YYYY-MM-DD", "count": int}

    Examples
    --------
    - NL: "summary of twitter" -> collection="twitter"
    - NL: "how big is mastodon collection?" -> collection="mastodon"
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
    Aggregate documents in a collection into time buckets.

    Endpoint
    --------
    GET /analysis/aggregate/collections/{collection}/aggregation

    Parameters
    ----------
    collection : {"Twitter", "Mastodon", "Reddit", "YouTube", "BlueSky", "Flickr"}
    Collection name.
    start_date : str (YYYY-MM-DD)
    Inclusive lower bound. If the user writes "2022-01 to 2022-06", assume
    start_date="2022-01-01".
    end_date : str (YYYY-MM-DD)
    Inclusive upper bound. If the user writes a year-month like "2022-06",
    assume the last day of that month (e.g., "2022-06-30").
    aggregation_level : {"day", "month", "year"}
    Granularity of aggregation.
    sentiment : bool, optional
    If True, return sentiment fields ("sentiment", "sentimentcount") where available.
    extra_params : dict, optional
    Additional query parameters passed through to the API.

    Returns
    -------
    list[CountPoint]
    Each item has at least {"time": str, "count": int}. If sentiment=True,
    may also include {"sentiment": float, "sentimentcount": int}.

    Strict Mapping Rules for LLMs
    -----------------------------
    - If the query mentions "twitter", set collection="twitter" (no follow-up question).
    - If the query mentions "aggregate by month" or "by month", prefer aggregate_month.
    - If the user provides a range like "2022-01 to 2022-06", expand to the exact
    month/day boundaries.

    Examples
    --------
    - NL: "twitter from 2022-01 to 2022-06 aggregate by month" ->
    aggregate_month(collection="twitter", start_date="2022-01-01", end_date="2022-06-30")
    - NL: "mastodon in 2023 by day" with no dates -> ask once for the date window.
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
    Aggregate by **day**.

    Wrapper of `aggregate_by_time` with aggregationLevel="day".

    Quick NL Mapping
    ----------------
    - Mentions of "by day" or "daily" should call this tool directly.

    Example
    -------
    - NL: "twitter daily counts for March 2022" ->
    aggregate_day(collection="twitter", start_date="2022-03-01", end_date="2022-03-31")
    """
    client = get_client()
    collection = collection.lower().strip()
    return client.aggregate_day(collection=collection, start_date=start_date, end_date=end_date,
                                sentiment=bool(sentiment), **(extra_params or {}),)

@mcp.tool()
def aggregate_month(collection: str, start_date: str, end_date: str, sentiment: bool = False,
                    extra_params: dict[str, Any] | None = None,) -> list[dict[str, Any]]:
    """
    Aggregate by **month**.

    Wrapper of `aggregate_by_time` with aggregationLevel="month". The AIO client
    may chunk long ranges and align to full month boundaries.

    Quick NL Mapping
    ----------------
    - Mentions of "by month", "monthly", or phrases like "aggregate by month"
    should call this tool.

    Example
    -------
    - NL: "want to know twitter count from 2022-01 to 2022-06 aggregate by month" ->
    aggregate_month(collection="twitter", start_date="2022-01-01", end_date="2022-06-30")
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
    Aggregate seasonal patterns: day-of-week or hour-of-day distribution.

    Endpoint
    --------
    GET /analysis/aggregate/collections/{collection}/seasonality

    Parameters
    ----------
    collection : {"Twitter", "Mastodon", "Reddit", "YouTube", "BlueSky", "Flickr"}
    start_date : str (YYYY-MM-DD)
    end_date : str (YYYY-MM-DD)
    aggregation_level : {"dayofweek", "hourofday"}
    Bucket type.
    sentiment : bool, optional
    extra_params : dict, optional

    Returns
    -------
    list of dict
    e.g., [{"bucket": "Monday", "count": 123}, ...]

    NL Examples
    -----------
    - "twitter 2022-01 to 2022-03 by day-of-week" -> aggregation_level="dayofweek"
    - "mastodon Jan 2023 by hour" -> aggregation_level="hourofday"
    """
    client = get_client()
    collection = collection.lower().strip()
    return client.aggregate_seasonality(collection=collection, start_date=start_date, end_date=end_date,
                                        aggregation_level=aggregation_level, sentiment=bool(sentiment),
                                        **(extra_params or {}),)

# Tool -- Language aggregation
@mcp.tool()
def aggregate_language(collection: str, start_date: str, end_date: str, aggregation_level: str, sentiment: bool = False,
                       extra_params: dict[str, Any] | None = None,) -> list[dict[str, Any]]:
    """
    Aggregate by language or geographic grouping.

    Endpoint
    --------
    GET /analysis/aggregate/collections/{collection}/language

    Parameters
    ----------
    collection : {"Twitter", "Mastodon", "Reddit", "YouTube", "BlueSky", "Flickr"}
    start_date : str (YYYY-MM-DD)
    end_date : str (YYYY-MM-DD)
    aggregation_level : {"language", "country", "state", "gccsa", "suburb"}
    sentiment : bool, optional
    extra_params : dict, optional

    Returns
    -------
    list of dict
    Each item has grouping key(s), counts, and optional sentiment fields.

    NL Examples
    -----------
    - "twitter Jan 2022 language breakdown" -> aggregation_level="language"
    - "twitter 2022-01 to 2022-06 by country" -> aggregation_level="country"
    """
    client = get_client()
    collection = collection.lower().strip()
    return client.aggregate_language(collection=collection, start_date=start_date, end_date=end_date, aggregation_level=aggregation_level,
                                     sentiment=bool(sentiment), extra_params=extra_params or {},)
# Tool -- Place aggregation
@mcp.tool()
def aggregate_place(collection: str, start_date: str, end_date: str, aggregation_level: str, sentiment: bool = False,
                    extra_params: dict[str, Any] | None = None,) -> list[dict[str, Any]]:
    """
    Aggregate by geographic location (country/state/GCCSA/suburb) or language.

    Endpoint
    --------
    GET /analysis/aggregate/collections/{collection}/place

    Parameters
    ----------
    collection : {"Twitter", "Mastodon", "Reddit", "YouTube", "BlueSky", "Flickr"}
    start_date : str (YYYY-MM-DD)
    end_date : str (YYYY-MM-DD)
    aggregation_level : {"country", "state", "gccsa", "suburb", "language"}
    sentiment : bool, optional
    extra_params : dict, optional

    Returns
    -------
    list of dict

    Notes
    -----
    Some deployments may expose the same set under /place; if your backend
    differentiates /language and /place, prefer the dedicated endpoint.
    """
    client = get_client()
    collection = collection.lower().strip()
    return client.aggregate_place(collection=collection, start_date=start_date, end_date=end_date, aggregation_level=aggregation_level,
                                  sentiment=bool(sentiment), extra_params=extra_params or {},)

# ------- Terms / Keywords -------
# Tool -- All terms (dictionary of stem → count)
@mcp.tool()
def aggregate_terms_all(collection: str, start_date: str, end_date: str, extra_params: dict[str, Any] | None = None,) -> dict[str, Any]:
    """
    Retrieve all term frequencies for a date window.

    Endpoint
    --------
    GET /analysis/aggregate/collections/{collection}/terms/all

    Parameters
    ----------
    collection : {"Twitter", "Mastodon", "Reddit", "YouTube", "BlueSky", "Flickr"}
    start_date : str (YYYY-MM-DD)
    end_date : str (YYYY-MM-DD)
    extra_params : dict, optional

    Returns
    -------
    dict
    {term: frequency, ...}

    Example
    -------
    - NL: "top terms in twitter May 2022" -> call this, sort on client if needed.
    """
    client = get_client()
    collection = collection.lower().strip()
    return client.aggregate_terms_all(collection=collection, start_date=start_date, end_date=end_date, extra_params=extra_params or {},)

# Tool -- Specific terms (daily series)
@mcp.tool()
def aggregate_terms_specific(collection: str, start_date: str, end_date: str, terms: list[str] | str,
                             extra_params: dict[str, Any] | None = None,) -> dict[str, Any]:
    """
    Retrieve daily frequency series for specific terms.

    Endpoint
    --------
    GET /analysis/aggregate/collections/{collection}/terms/specific

    Parameters
    ----------
    collection : {"Twitter", "Mastodon", "Reddit", "YouTube", "BlueSky", "Flickr"}
    start_date : str (YYYY-MM-DD)
    end_date : str (YYYY-MM-DD)
    terms : list[str] or comma-separated str
    extra_params : dict, optional

    Returns
    -------
    dict
    { term: [{"time": "YYYY-MM-DD", "count": int}, ...], ... }

    Examples
    --------
    - NL: "twitter daily series for ['covid','vaccine'] in 2022-01" ->
    aggregate_terms_specific(collection="twitter", start_date="2022-01-01",
    end_date="2022-01-31", terms=["covid","vaccine"]).
    """
    client = get_client()
    collection = collection.lower().strip()
    return client.aggregate_terms_specific(collection=collection, start_date=start_date, end_date=end_date, terms=terms,
                                           extra_params=extra_params or {},)
# ------- NLP / Embeddings / Topics -------
# Tool -- List terms available for a given day’s embedding model
@mcp.tool()
def nlp_terms_available(collection: str, day: str, extra_params: dict[str, Any] | None = None,
                        ) -> list[str] | dict[str, Any]:
    """
    List terms available in the embedding model for a given day.

    Endpoint
    --------
    GET /analysis/nlp/collections/{collection}/terms/available

    Parameters
    ----------
    collection : {"Twitter", "Mastodon", "Reddit", "YouTube", "BlueSky", "Flickr"}
    day : str (YYYY-MM-DD)
    extra_params : dict, optional

    Returns
    -------
    list[str] | dict
    A list of terms or a dict with metadata depending on backend.

    Example
    -------
    - NL: "available embedding terms for twitter on 2022-03-15".
    """
    client = get_client()
    collection = collection.lower().strip()
    return client.nlp_terms_available(collection=collection, day=day, extra_params=extra_params or {},)

# Tool -- Similar terms for a query word on a given day
@mcp.tool()
def nlp_term_similarity(collection: str, day: str, term: str, topk: int | None = None,
                        extra_params: dict[str, Any] | None = None,) -> dict[str, float]:
    """
    Retrieve top-K most similar terms for a given word and date.

    Endpoint
    --------
    GET /analysis/nlp/collections/{collection}/terms/similarity

    Parameters
    ----------
    collection : {"Twitter", "Mastodon", "Reddit", "YouTube", "BlueSky", "Flickr"}
    day : str (YYYY-MM-DD)
    term : str
    topk : int, optional (alias: topK)
    extra_params : dict, optional

    Returns
    -------
    dict
    { similar_term: similarity_score, ... }

    Notes for LLM
    -------------
    - If user says "top 10 similar words to X on 2022-05-21", set topk=10.
    - Preserve case of the query term where possible.
    """
    client = get_client()
    collection = collection.lower().strip()
    kwargs = dict(extra_params or {})
    if topk is not None:
        kwargs.setdefault("topK", int(topk))
    return client.nlp_term_similarity(collection=collection, day=day, term=term, extra_params=kwargs,)

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