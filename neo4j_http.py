"""Minimal Neo4j HTTPS Query API (v2) client using `requests`.

Credentials come from environment variables / .env. Nothing secret is printed.
Run `python neo4j_http.py` to test the connection.
"""
import os
import re
import sys
import time

import requests

import common  # noqa: F401  (loads .env)


class Neo4jError(RuntimeError):
    pass


class Result:
    def __init__(self, rows, counters):
        self.rows = rows
        self.counters = counters or {}


def _plain(v):
    """Unwrap typed Query-API values ({"$type":..., "_value":...}) into plain Python."""
    if isinstance(v, dict):
        if "$type" in v and "_value" in v:
            t, val = v["$type"], v["_value"]
            if t == "Integer":
                return int(val)
            if t == "Float":
                return float(val)
            return _plain(val)
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_plain(x) for x in v]
    return v


class Neo4jQueryAPI:
    def __init__(self, max_retries: int = 5, timeout: int = 120):
        self.url = os.environ.get("NEO4J_QUERY_API_URL", "").strip()
        user = os.environ.get("NEO4J_USERNAME", "neo4j").strip()
        password = os.environ.get("NEO4J_PASSWORD", "")
        if not self.url or not password:
            raise Neo4jError(
                "Set NEO4J_QUERY_API_URL and NEO4J_PASSWORD (see .env.example). "
                "Use the exact Query API URL from the Aura console."
            )
        if not self.url.lower().startswith("https://"):
            raise Neo4jError("NEO4J_QUERY_API_URL must be an https:// URL.")
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = requests.Session()
        self.session.auth = (user, password)
        self.session.headers.update({"Content-Type": "application/json", "Accept": "application/json"})

    def run(self, statement: str, parameters: dict | None = None) -> Result:
        body = {"statement": statement, "includeCounters": True}
        if parameters:
            body["parameters"] = parameters
        delay, last = 1.5, "unknown error"
        for attempt in range(1, self.max_retries + 1):
            try:
                r = self.session.post(self.url, json=body, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as e:
                last = f"network error ({type(e).__name__})"
            else:
                if r.status_code == 401:
                    raise Neo4jError("Authentication failed (HTTP 401). Check NEO4J_USERNAME / NEO4J_PASSWORD.")
                try:
                    payload = r.json()
                except ValueError:
                    payload = {}
                errors = payload.get("errors") if isinstance(payload, dict) else None
                if r.status_code in (429, 500, 502, 503, 504):
                    last = f"HTTP {r.status_code}"
                elif r.status_code >= 400 or errors:
                    code = (errors or [{}])[0].get("code", f"HTTP {r.status_code}")
                    msg = (errors or [{}])[0].get("message", r.text[:300])
                    if str(code).startswith("Neo.TransientError"):
                        last = f"{code}: {msg}"
                    else:
                        raise Neo4jError(f"{code}: {msg}")
                else:
                    data = (payload.get("data") or {}) if isinstance(payload, dict) else {}
                    fields, values = data.get("fields", []), data.get("values", [])
                    rows = [dict(zip(fields, _plain(v))) for v in values]
                    return Result(rows, payload.get("counters"))
            if attempt < self.max_retries:
                print(f"  transient failure ({last}); retry {attempt}/{self.max_retries - 1} in {delay:.0f}s")
                time.sleep(delay)
                delay = min(delay * 2, 30)
        raise Neo4jError(f"Gave up after {self.max_retries} attempts: {last}")

    def version(self):
        r = self.run("CALL dbms.components() YIELD name, versions, edition RETURN name, versions, edition")
        row = r.rows[0]
        return row["versions"][0], row["edition"]


def version_supports_vector(version: str) -> bool:
    """Vector indexes need Neo4j 5.11+ (or calendar versioning 2025.x+)."""
    m = re.match(r"(\d+)\.(\d+)", version)
    if not m:
        return True  # unknown format: let the server decide
    major, minor = int(m.group(1)), int(m.group(2))
    return major != 5 or minor >= 11


if __name__ == "__main__":
    try:
        c = Neo4jQueryAPI()
        v, ed = c.version()
        print(f"Connected. Neo4j {v} ({ed}); vector-index capable: {version_supports_vector(v)}")
        n = c.run("MATCH (a:Act) RETURN count(a) AS n").rows[0]["n"]
        print(f"Existing :Act nodes: {n}")
    except Neo4jError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
