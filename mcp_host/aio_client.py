import os
import time
import requests
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from json import JSONDecodeError


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
    def aggregate_by_time(self, collection: str, start_date: str, end_date: str, aggregation_level: str,  # "day" | "month" | "year"
        sentiment: bool = False, extra_params: dict | None = None,):
        """
        GET /analysis/aggregate/collections/{collection}/aggregation
        :param：
          - startDate, endDate: "YYYY-MM-DD" (OR date-time)
          - aggregationLevel: "day" | "month" | "year"
          - sentiment:
                if False, then return count
                if True, then return sentiment and sentimentcount
        :return
          - count: [{"time":"2021-07-01","count":N}, ...]
          - sentiment: [{"time":"2021-07-01","sentiment":S,"sentimentcount":N}, ...]
        """
        level = aggregation_level.lower()
        if level not in {"day", "month", "year"}:
            raise ValueError("aggregation_level must be one of: day, month, year")

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

        path = f"/analysis/aggregate/collections/{collection}/aggregation"
        return self._get(path, params=params)


# --- login test ---
if __name__ == "__main__":
    client = AIOClient()
    version = client._get("/version")
    print("API version:", version)
