"""Upload every document in sample_docs/ through the public API and wait for `ready`.

Usage: uv run python scripts/ingest_samples.py [--api-url http://localhost:8000]
"""

import argparse
import os
import sys
import time
from pathlib import Path

import httpx
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "sample_docs"


def api_key() -> str:
    keys = os.environ.get("API_KEYS") or dotenv_values(ROOT / ".env").get("API_KEYS") or ""
    key = keys.split(",")[0].strip()
    if not key:
        sys.exit("No API key: set API_KEYS in .env")
    return key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()

    files = sorted(p for p in SAMPLES.iterdir() if p.is_file() and not p.name.startswith("."))
    headers = {"X-API-Key": api_key()}
    with httpx.Client(base_url=args.api_url, headers=headers, timeout=60) as client:
        pending: dict[str, str] = {}
        for path in files:
            resp = client.post("/v1/documents", files={"file": (path.name, path.read_bytes())})
            resp.raise_for_status()
            doc = resp.json()
            verb = "uploaded" if resp.status_code == 202 else "already ingested"
            print(f"{verb:>16}: {path.name} ({doc['id']})")
            pending[doc["id"]] = path.name

        deadline = time.monotonic() + args.timeout
        results: dict[str, dict[str, object]] = {}
        while pending and time.monotonic() < deadline:
            for doc_id in list(pending):
                doc = client.get(f"/v1/documents/{doc_id}").json()
                if doc["status"] != "processing":
                    results[pending.pop(doc_id)] = doc
            if pending:
                time.sleep(1)

    failed = False
    for name, doc in sorted(results.items()):
        print(f"{doc['status']:>8}: {name}  chunks={doc['num_chunks']}  {doc.get('error') or ''}")
        failed |= doc["status"] != "ready"
    for name in pending:
        print(f" timeout: {name}")
    if failed or pending:
        sys.exit(1)
    print(f"All {len(results)} documents ready.")


if __name__ == "__main__":
    main()
