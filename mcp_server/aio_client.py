import os
import time
from datetime import datetime, timedelta
import calendar
from json import JSONDecodeError

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from requests.auth import HTTPBasicAuth
from urllib3.util.retry import Retry

class AIOClient:
    MAX_DAYS_PER_CALL = 130

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

    @staticmethod
    def _date_chunks(start_ymd: str, end_ymd: str, max_days: int):
        s = datetime.fromisoformat(start_ymd[:10])
        e = datetime.fromisoformat(end_ymd[:10])
        one_day = timedelta(days=1)
        cur = s
        while cur <= e:
            chunk_end = min(cur + timedelta(days=max_days - 1), e)
            yield cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
            cur = chunk_end + one_day

    @staticmethod
    def _sum_inplace(target: dict, src: dict, keys=("count", "sentiment", "sentimentcount")):
        for k in keys:
            if k in src:
                if k not in target:
                    target[k] = 0 if k != "sentiment" else 0.0
                target[k] += src.get(k, 0)

    @staticmethod
    def _merge_seasonality_rows(acc: dict, rows: list, sentiment: bool):
        if not rows:
            return
        sample = rows[0]
        if "time" in sample:
            key_field = "time"
        elif "dayofweek" in sample:
            key_field = "dayofweek"
        elif "hourofday" in sample:
            key_field = "hourofday"
        else:
            key_field = "time"

        for r in rows:
            k = r.get(key_field)
            if k not in acc:
                acc[k] = dict(r)
                acc[k].setdefault("count", 0)
                acc[k].setdefault("sentiment", 0.0)
                acc[k].setdefault("sentimentcount", 0)
            else:
                if "count" in r:
                    acc[k]["count"] += r.get("count", 0)
                if sentiment:
                    acc[k]["sentiment"] += r.get("sentiment", 0.0)
                    acc[k]["sentimentcount"] += r.get("sentimentcount", 0)

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
                              sentiment: bool = False, extra_params: dict | None = None):
        level = aggregation_level.lower()
        if level not in {"dayofweek", "hourofday"}:
            raise ValueError("aggregation_level must be one of: dayofweek, hourofday")

        start_date, end_date = self._clamp_to_window(collection, start_date, end_date)

        base_params = {
            "startDate": None,
            "endDate": None,
            "aggregationLevel": level,
        }
        if sentiment:
            base_params["sentiment"] = "true"
        if extra_params:
            base_params.update(extra_params)

        path = f"/analysis/aggregate/collections/{collection}/seasonality"
        acc = {}
        for chunk_start, chunk_end in self._date_chunks(start_date[:10], end_date[:10], max_days=130):
            params = dict(base_params)
            params["startDate"] = chunk_start
            params["endDate"] = chunk_end

            rows = self._get(path, params=params)
            if isinstance(rows, list):
                self._merge_seasonality_rows(acc, rows, sentiment)
            else:
                pass
        def _sort_key(k):
            try:
                return int(k)
            except Exception:
                return k
        merged = [acc[k] for k in sorted(acc.keys(), key=_sort_key)]
        return merged

    def aggregate_language(self, collection: str, start_date: str, end_date: str, aggregation_level: str, sentiment: bool = False, extra_params: dict | None = None,):
        level = aggregation_level.lower()
        if level not in {"language", "country", "state", "gccsa", "suburb"}:
            raise ValueError("aggregation_level must be one of: language, country, state, gccsa, suburb")

        start_date, end_date = self._clamp_to_window(collection, start_date, end_date)
        base_params = {"startDate": None, "endDate": None,  "aggregationLevel": level,
        }
        if sentiment:
            base_params["sentiment"] = "true"
        if extra_params:
            base_params.update(extra_params)

        path = f"/analysis/language/collections/{collection}"
        acc = {}
        key_field = level
        for cs, ce in self._date_chunks(start_date[:10], end_date[:10], self.MAX_DAYS_PER_CALL):
            params = dict(base_params)
            params["startDate"] = cs
            params["endDate"] = ce
            rows = self._get(path, params=params)
            for r in rows:
                if key_field not in r:
                    continue
                k = r[key_field]
                if k not in acc:
                    acc[k] = {key_field: k}
                self._sum_inplace(acc[k], r, keys=("count", "sentiment", "sentimentcount"))
        merged = [acc[k] for k in sorted(acc.keys())]
        return merged

    def aggregate_place(self, collection: str, start_date: str, end_date: str, aggregation_level: str, sentiment: bool = False, extra_params: dict | None = None,):
        level = aggregation_level.lower()
        if level not in {"country", "state", "gccsa", "suburb", "language"}:
            raise ValueError("aggregation_level must be one of: country, state, gccsa, suburb, language")

        start_date, end_date = self._clamp_to_window(collection, start_date, end_date)
        base_params = {"startDate": None, "endDate": None, "aggregationLevel": level,}
        if sentiment:
            base_params["sentiment"] = "true"
        if extra_params:
            base_params.update(extra_params)
        path = f"/analysis/place/collections/{collection}"
        acc = {}
        key_field = level
        for cs, ce in self._date_chunks(start_date[:10], end_date[:10], self.MAX_DAYS_PER_CALL):
            params = dict(base_params)
            params["startDate"] = cs
            params["endDate"] = ce

            rows = self._get(path, params=params)
            for r in rows:
                if key_field not in r:
                    for alt in (key_field, "name", "place", "region"):
                        if alt in r:
                            key_field = alt
                            break
                if key_field not in r:
                    continue
                k = r[key_field]
                if k not in acc:
                    acc[k] = {key_field: k}
                self._sum_inplace(acc[k], r, keys=("count", "sentiment", "sentimentcount"))
            merged = [acc[k] for k in sorted(acc.keys())]
            return merged

    def aggregate_terms_all(self, collection: str, start_date: str, end_date: str, extra_params: dict | None = None,) -> dict:
        """
        Fetch all stem terms and their aggregated counts within the date window.
        Returns a dict like: { "terms": { "<stem>": <count>, ... } }
        """
        collection = (collection or "").strip().lower()
        start_date, end_date = self._clamp_to_window(collection, start_date, end_date)

        base_params = {"startDate": None, "endDate": None,}
        if extra_params:
            base_params.update(extra_params)

        path = f"/analysis/terms/collections/{collection}"

        acc_terms: dict[str, int] = {}
        for cs, ce in self._date_chunks(start_date[:10], end_date[:10], self.MAX_DAYS_PER_CALL):
            params = dict(base_params)
            params["startDate"] = cs
            params["endDate"] = ce

        res = self._get(path, params=params)
        for term, cnt in res["terms"].items():
            acc_terms[term] = acc_terms.get(term, 0) + int(cnt or 0)

        return {"terms": acc_terms}

    def aggregate_terms_specific(self, collection: str, start_date: str, end_date: str, terms: list[str] | str, extra_params: dict | None = None,) -> dict:
        def _norm_terms(ts):
            if isinstance(ts, (list, tuple)):
                return [str(t).strip().lower() for t in ts if str(t).strip()]
            parts = []
            for ch in str(ts).split(","):
                parts.extend(ch.split())
            return [p.strip().lower() for p in parts if p.strip()]

        def _parse_payload(payload):
            import json
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except Exception:
                    return {}
            if isinstance(payload, dict):
                return payload
            if isinstance(payload, list):
                out = {}
                for r in payload:
                    if not isinstance(r, dict):
                        continue
                    t = str(r.get("term", "")).strip().lower()
                    d = str(r.get("date") or r.get("time") or "").strip()
                    c = int(r.get("count", 0) or 0)
                    if t and d:
                        out.setdefault(t, []).append({"date": d, "count": c})
                return out
            return {}

        collection = (collection or "").strip().lower()
        start_date, end_date = self._clamp_to_window(collection, start_date, end_date)

        term_list = _norm_terms(terms)
        if not term_list:
            return {}

        base_params = {
            "terms": ",".join(term_list),
            "startDate": None,
            "endDate": None,
        }
        if extra_params:
            base_params.update(extra_params)

        path = f"/analysis/terms/collections/{collection}/term"

        from collections import defaultdict
        acc: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for cs, ce in self._date_chunks(start_date[:10], end_date[:10], self.MAX_DAYS_PER_CALL):
            params = dict(base_params)
            params["startDate"] = cs
            params["endDate"] = ce

            payload = self._get(path, params=params)
            parsed = _parse_payload(payload)

            for term, series in parsed.items():
                tkey = str(term).strip().lower()
                if not isinstance(series, list):
                    continue
                for row in series:
                    d = str(row.get("date") or row.get("time") or "").strip()
                    if not d:
                        continue
                    try:
                        from datetime import datetime
                        dcanon = datetime.fromisoformat(d[:10]).strftime("%Y-%m-%d")
                    except Exception:
                        try:
                            y, m, dd = d[:10].split("-")
                            dcanon = f"{int(y):04d}-{int(m):02d}-{int(dd):02d}"
                        except Exception:
                            continue
                    c = int(row.get("count", 0) or 0)
                    acc[tkey][dcanon] += c

        out: dict[str, list[dict]] = {}
        for term in sorted(acc.keys()):
            dates = sorted(acc[term].keys())
            out[term] = [{"date": d, "count": acc[term][d]} for d in dates]
        return out

    def nlp_terms_available(self, collection: str, day: str, extra_params: dict | None = None,) -> list[str] | dict:
        """
        List terms available for querying in the word2vec model of a given day.
        Returns either a list[str] or a dict like {"terms": [...]} depending on server.
        """
        collection = (collection or "").strip().lower()
        day = day[:10]
        params = {}
        if extra_params:
            params.update(extra_params)
        path = f"/analysis/nlp/collections/{collection}/days/{day}/terms"
        return self._get(path, params=params)

    def nlp_term_similarity(self, collection: str, day: str, term: str, topk: int | None = None,
                            extra_params: dict | None = None,) -> dict[str, float]:
        """
        Query top similar terms for `term` on a given day.
        Returns a mapping { similar_term: cosine_similarity, ... }.
        If server supports a topK query param, pass via extra_params or topk.
        """
        collection = (collection or "").strip().lower()
        day = day[:10]
        term = (term or "").strip().lower()

        params = {}
        if topk is not None:
            params["topK"] = int(topk)
        if extra_params:
            params.update(extra_params)

        path = f"/analysis/nlp/collections/{collection}/days/{day}/terms/{term}"
        return self._get(path, params=params)

# --- test ---
if __name__ == "__main__":
    import json
    def phead(title, data, n=10):
        print(f"\n=== {title} (showing first {n}) ===")
        if isinstance(data, list):
            print(json.dumps(data[:n], ensure_ascii=False, indent=2))
            print(f"... total {len(data)} rows")
        else:
            print(json.dumps(data, ensure_ascii=False, indent=2))

    def ppreview(title, obj, n=10):
        print(f"\n=== {title} ===")
        if isinstance(obj, list):
            print(json.dumps(obj[:n], ensure_ascii=False, indent=2))
            print(f"... total {len(obj)} terms")
        elif isinstance(obj, dict) and "terms" in obj and isinstance(obj["terms"], list):
            print(json.dumps(obj["terms"][:n], ensure_ascii=False, indent=2))
            print(f"... total {len(obj['terms'])} terms")
        elif isinstance(obj, dict):
            items = sorted(obj.items(), key=lambda kv: kv[1], reverse=True)[:n]
            print(json.dumps({k: v for k, v in items}, ensure_ascii=False, indent=2))
            print(f"... total {len(obj)} neighbors")
        else:
            print(json.dumps(obj, ensure_ascii=False, indent=2))
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
    # dow_counts = client.aggregate_seasonality("twitter", "2021-07-01", "2022-07-11", "dayofweek")
    # print("Day-of-week counts:", dow_counts)
    # hod_sent = client.aggregate_seasonality("twitter", "2021-07-01", "2022-07-11", "hourofday", sentiment=True)
    # print("Hour-of-day sentiment:", hod_sent)
    # lang_lang = client.aggregate_language("twitter", "2021-07-01", "2022-07-11", aggregation_level="language")
    # phead("Language aggregation (level=language, count)", lang_lang, n=10)
    # lang_state = client.aggregate_language("twitter", "2021-07-27", "2021-12-31", aggregation_level="state")
    # phead("Language aggregation (level=state, count)", lang_state, n=10)
    # place_country = client.aggregate_place("twitter", "2021-07-27", "2021-12-31", aggregation_level="country")
    # phead("Place aggregation (level=country, count)", place_country, n=10)
    # place_suburb_sent = client.aggregate_place("twitter", "2021-07-27", "2021-12-31", aggregation_level="suburb", sentiment=True)
    # phead("Place aggregation (level=suburb, sentiment)", place_suburb_sent, n=10)
    # terms_all = client.aggregate_terms_all("twitter", "2021-07-27", "2021-12-31")
    # phead("All terms (twitter, 2021-07-27 ~ 2021-12-31)", terms_all, n=10)
    # terms_spec = client.aggregate_terms_specific("twitter","2021-07-27","2021-12-31",terms=["ukrain"])
    # phead("Specific term 'ukrain' daily counts", terms_spec.get("ukrain", []), n=10)
    model_terms = client.nlp_terms_available("twitter", "2021-07-09")
    ppreview("NLP terms available (twitter, 2021-07-09)", model_terms, n=20)
    sim_map = client.nlp_term_similarity("twitter", "2021-07-09", term="vaccin", topk=25)
    ppreview("NLP term similarity (vaccin, 2021-07-09)", sim_map, n=15)
