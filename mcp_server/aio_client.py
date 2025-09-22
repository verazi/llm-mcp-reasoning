import os
import time
from datetime import datetime
import calendar
from json import JSONDecodeError

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from requests.auth import HTTPBasicAuth
from urllib3.util.retry import Retry


class AIOClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, timeout: int = 30):
        load_dotenv()

        self.base_url = base_url or os.getenv("AIO_BASE_URL", "https://api.aio.eresearch.unimelb.edu.au")
        self.api_key = api_key or os.getenv("AIO_API_KEY")
        if not self.api_key:
            raise ValueError("API key not provided and AIO_API_KEY env var not set")

        self.timeout = timeout
        self.jwt = None
        self.jwt_time = 0
        self.jwt_ttl = 24 * 3600  # 24 hours refresh

        self.session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session.mount("http://", HTTPAdapter(max_retries=retry))

    # ---------- JWT ----------
    def _refresh_jwt(self):
        url = f"{self.base_url}/login"
        res = self.session.post(url, auth=HTTPBasicAuth("apikey", self.api_key), timeout=self.timeout)
        res.raise_for_status()
        self.jwt = res.text.strip()
        self.jwt_time = time.time()
        return self.jwt

    def _get_jwt(self):
        if not self.jwt or (time.time() - self.jwt_time) > self.jwt_ttl:
            self._refresh_jwt()
        return self.jwt

    def _auth_headers(self):
        return {"Authorization": f"Bearer {self._get_jwt()}"}

    # ---------- helpers ----------
    @staticmethod
    def _maybe_json(res: requests.Response):
        try:
            return res.json()
        except (ValueError, JSONDecodeError):
            return res.text.strip()

    def _request_with_auto_relogin(self, method: str, path: str, **kwargs):
        url = f"{self.base_url}{path}"
        headers = kwargs.pop("headers", {})
        headers.update(self._auth_headers())
        try:
            res = self.session.request(method, url, headers=headers, timeout=self.timeout, **kwargs)
            if res.status_code == 401: # refresh
                self._refresh_jwt()
                headers["Authorization"] = f"Bearer {self.jwt}"
                res = self.session.request(method, url, headers=headers, timeout=self.timeout, **kwargs)
            res.raise_for_status()
            return res
        except requests.RequestException as e:
            raise

    def _clamp_to_window(self, collection: str, start_date: str, end_date: str) -> tuple[str, str]:
        s = self.collection_summary(collection)  # {"startDate": "...", "endDate": "..."}
        lo, hi = s["startDate"][:10], s["endDate"][:10]
        start = max(start_date[:10], lo)
        end = min(end_date[:10], hi)
        return start, end

    # ------- public -------
    def _get(self, endpoint: str, **kwargs):
        res = self._request_with_auto_relogin("GET", endpoint, **kwargs)
        return self._maybe_json(res)

    def _post(self, endpoint: str, json=None, **kwargs):
        res = self._request_with_auto_relogin("POST", endpoint, json=json, **kwargs)
        return self._maybe_json(res)

    # ------- Summary of a Collection -------
    def collection_summary(self, collection: str, params: dict | None = None):
        path = f"/data/collections/{collection}/summary"
        return self._get(path, params=params or {})

    # ------- Aggregation -------
    @staticmethod
    def _align_month_start(date_str: str) -> str:
        dt = datetime.fromisoformat(date_str[:10])
        return f"{dt.year:04d}-{dt.month:02d}-01"

    @staticmethod
    def _month_end(date_str: str) -> str:
        dt = datetime.fromisoformat(date_str[:10])
        last = calendar.monthrange(dt.year, dt.month)[1]
        return f"{dt.year:04d}-{dt.month:02d}-{last}"

    @staticmethod
    def _add_months(date_str: str, months: int) -> str:
        dt = datetime.fromisoformat(date_str[:10])
        y = dt.year + (dt.month - 1 + months) // 12
        m = (dt.month - 1 + months) % 12 + 1
        d = min(dt.day, calendar.monthrange(y, m)[1])
        return f"{y:04d}-{m:02d}-{d:02d}"

    @staticmethod
    def _months_between_inclusive(start_ymd: str, end_ymd: str) -> int:
        s = datetime.fromisoformat(start_ymd[:10])
        e = datetime.fromisoformat(end_ymd[:10])
        return (e.year - s.year) * 12 + (e.month - s.month) + 1

    def aggregate_by_time(self, collection: str, start_date: str, end_date: str, aggregation_level: str,  # "day" | "month" | "year"
        sentiment: bool = False, extra_params: dict | None = None,):
        level = aggregation_level.lower()
        if level not in {"day", "month", "year"}:
            raise ValueError("aggregation_level must be one of: day, month, year")

        start_date, end_date = self._clamp_to_window(collection, start_date, end_date)
        if level == "month":
            start_date = self._align_month_start(start_date)
            end_date = self._month_end(end_date)

        params = {
            "startDate": start_date[:10],
            "endDate": end_date[:10],
            "aggregationLevel": level,
        }
        if sentiment:
            params["sentiment"] = "true"
        if extra_params:
            params.update(extra_params)

        path = f"/analysis/aggregate/collections/{collection}/aggregation"

        if level == "month":
            MAX_MONTHS_PER_CALL = 4
            total_months = self._months_between_inclusive(start_date, end_date)

            if total_months <= MAX_MONTHS_PER_CALL:
                return self._get(path, params=params)

            out = {}
            chunk_start = start_date
            while True:
                chunk_end_month_start = self._add_months(chunk_start, MAX_MONTHS_PER_CALL - 1)
                chunk_end = self._month_end(chunk_end_month_start)

                if datetime.fromisoformat(chunk_end) > datetime.fromisoformat(end_date):
                    chunk_end = end_date

                chunk_params = dict(params)
                chunk_params["startDate"] = chunk_start[:10]
                chunk_params["endDate"] = chunk_end[:10]

                chunk_res = self._get(path, params=chunk_params)
                for row in chunk_res:
                    key = row["time"]
                    out[key] = row

                if datetime.fromisoformat(chunk_end) >= datetime.fromisoformat(end_date):
                    break

                next_month_start = self._add_months(chunk_start, MAX_MONTHS_PER_CALL)
                chunk_start = self._align_month_start(next_month_start)

            merged = [out[k] for k in sorted(out.keys())]
            return merged

        if level == "year":
            merged_by_year = {}
            chunk_start = self._align_month_start(start_date)
            while True:
                chunk_end = self._month_end(chunk_start)
                if datetime.fromisoformat(chunk_end) > datetime.fromisoformat(end_date):
                    chunk_end = end_date

                chunk_params = dict(params)
                chunk_params["startDate"] = chunk_start[:10]
                chunk_params["endDate"] = chunk_end[:10]

                chunk_res = self._get(path, params=chunk_params)
                for row in chunk_res:
                    y = row["time"]
                    if y not in merged_by_year:
                        merged_by_year[y] = dict(row)
                    else:
                        if sentiment:
                            merged_by_year[y]["sentiment"] += row.get("sentiment", 0.0)
                            merged_by_year[y]["sentimentcount"] += row.get("sentimentcount", 0)
                        else:
                            merged_by_year[y]["count"] += row.get("count", 0)

                if datetime.fromisoformat(chunk_end) >= datetime.fromisoformat(end_date):
                    break

                next_month_start = self._add_months(chunk_start, 1)
                chunk_start = self._align_month_start(next_month_start)

            return [merged_by_year[k] for k in sorted(merged_by_year.keys())]

        return self._get(path, params=params)

    def aggregate_day(self, collection: str, start_date: str, end_date: str, sentiment: bool = False, **kwargs):
        return self.aggregate_by_time(collection, start_date, end_date, "day", sentiment, extra_params=kwargs)

    def aggregate_month(self, collection: str, start_date: str, end_date: str, sentiment: bool = False, **kwargs):
        return self.aggregate_by_time(collection, start_date, end_date, "month", sentiment, extra_params=kwargs)

    def aggregate_year(self, collection: str, start_date: str, end_date: str, sentiment: bool = False, **kwargs):
        return self.aggregate_by_time(collection, start_date, end_date, "year", sentiment, extra_params=kwargs)

    def aggregate_seasonality(self, collection: str, start_date: str, end_date: str,
                              aggregation_level: str,  # "dayofweek" | "hourofday"
                              sentiment: bool = False, extra_params: dict | None = None,):
        level = aggregation_level.lower()
        if level not in {"dayofweek", "hourofday"}:
            raise ValueError("aggregation_level must be one of: dayofweek, hourofday")

        start_date, end_date = self._clamp_to_window(collection, start_date, end_date)

        params = {
            "startDate": start_date[:10],
            "endDate": end_date[:10],
            "aggregationLevel": level,
        }
        if sentiment:
            params["sentiment"] = "true"
        if extra_params:
            params.update(extra_params)

        path = f"/analysis/aggregate/collections/{collection}/seasonality"
        return self._get(path, params=params)


# --- test ---
if __name__ == "__main__":
    client = AIOClient()
    # version = client._get("/version")
    # print("API version:", version)
    # day_counts = client.aggregate_day("twitter", "2022-07-01", "2022-07-05")
    # print("Daily counts:", day_counts)

    # month_counts = client.aggregate_month("twitter", "2022-01-01", "2022-06-30")
    # print("Monthly count:", month_counts)
    # month_sent = client.aggregate_month("twitter", "2022-05-15", "2022-12-20", sentiment=True)
    # print("Monthly sentiment:", month_sent)

    # year_counts = client.aggregate_year("twitter", "2022-05-01", "2022-05-31")
    # print("Yearly counts:", year_counts)
    # year_sent = client.aggregate_year("twitter", "2022-06-30", "2023-07-03", sentiment=True)
    # print("Yearly sentiment:", year_sent)
    dow_counts = client.aggregate_seasonality("twitter", "2021-07-01", "2021-07-11", "dayofweek")
    print("Day-of-week counts:", dow_counts)

    hod_sent = client.aggregate_seasonality("twitter", "2021-07-01", "2021-07-11", "hourofday", sentiment=True)
    print("Hour-of-day sentiment:", hod_sent)