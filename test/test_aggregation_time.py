# tests/test_aggregation_time.py
from mcp_host.aio_client import AIOClient

def main():
    client = AIOClient()

    print("=== Count by day ===")
    res = client.aggregate_by_time(
        collection="youtube",
        start_date="2025-07-01",
        end_date="2025-07-11",
        aggregation_level="day",
        sentiment=False,
    )
    print(res[:5])

    print("=== Sentiment by day ===")
    res2 = client.aggregate_by_time(
        collection="twitter",
        start_date="2021-07-01",
        end_date="2021-07-11",
        aggregation_level="day",
        sentiment=True,
    )
    print(res2[:5])

if __name__ == "__main__":
    main()
