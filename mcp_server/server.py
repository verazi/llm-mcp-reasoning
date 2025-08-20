import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import logging
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession
from mcp_server.aio_client import AIOClient
from typing import Optional, Any

logging.basicConfig(level=logging.INFO, stream=sys.stderr)

# create FastMCP server
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
# Tool -- Aggregate by time (count / sentiment)
# Tool -- Seasonality aggregation
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



if __name__ == "__main__":
    mcp.run()
    # uv run mcp dev mcp_server/server.py
