"""
feed_ingest.py — push raw multi-cloud provider logs into the running app's
/api/v1/ingest/raw endpoint, to drive the live pipeline (ML -> graph -> risk ->
faithfulness-gated LLM narrative -> Prometheus/Loki -> Grafana).

The generated attack splits (Datasets/*/*_attack.json) are already in raw
provider format, so they can be posted straight through; the parser auto-detects
AWS/Azure/GCP per record.

Usage:
  python feed_ingest.py                         # one pass of Datasets/attacks_slow
  python feed_ingest.py --split Datasets/attacks_cal
  python feed_ingest.py --loop --interval 5     # keep feeding (moving dashboards)
"""
import argparse
import glob
import json
import os
import time

import requests


def feed_once(url: str, split: str) -> None:
    files = sorted(glob.glob(os.path.join(split, "*_attack.json")))
    if not files:
        raise SystemExit(f"No *_attack.json files in {split}")
    for f in files:
        logs = json.load(open(f))
        try:
            r = requests.post(f"{url}/api/v1/ingest/raw", json=logs, timeout=30)
            print(f"{os.path.basename(f):24s} {r.status_code} {r.json()}")
        except requests.RequestException as e:
            print(f"{os.path.basename(f):24s} ERROR {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--split", default="Datasets/attacks_slow")
    ap.add_argument("--loop", action="store_true", help="feed repeatedly")
    ap.add_argument("--interval", type=float, default=5.0, help="seconds between passes")
    args = ap.parse_args()

    if args.loop:
        print(f"Looping feed of {args.split} every {args.interval}s (Ctrl+C to stop)...")
        while True:
            feed_once(args.url, args.split)
            time.sleep(args.interval)
    else:
        feed_once(args.url, args.split)


if __name__ == "__main__":
    main()
