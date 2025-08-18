

# # --- tool 4: seasonality（dayofweek / hourofday）---
# @mcp.tool()
# def seasonality(
#     collection: Literal["twitter", "mastodon", "bluesky", "reddit", "youtube", "flickr"],
#     startDate: str,                                  # YYYY-MM-DD
#     endDate: str,                                    # YYYY-MM-DD
#     aggregationLevel: Literal["dayofweek", "hourofday"],
#     sentiment: bool = False,
# ):
#     """Seasonality aggregation by day-of-week or hour-of-day."""
#     return client.seasonality(
#         collection=collection,
#         start_date=startDate[:10],
#         end_date=endDate[:10],
#         aggregation_level=aggregationLevel,
#         sentiment=sentiment,
#     )

