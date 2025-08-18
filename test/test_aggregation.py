# tests/test_aggregation.py
import os
from mcp_host.aio_client import AIOClient

def main():
    client = AIOClient(
        base_url=os.getenv("AIO_BASE_URL", "https://api.aio.eresearch.unimelb.edu.au")
    )

    collections = [
        "twitter", "mastodon", "bluesky", "reddit", "youtube", "flickr"
    ]

    for c in collections:
        try:
            res = client.collection_summary(c)
            print(f"success {c:8s} -> {res}")
        except Exception as e:
            print(f"fail {c:8s} -> {e}")

if __name__ == "__main__":
    main()