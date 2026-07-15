"""
Pushes all dashboard JSON files in grafana/dashboards/ to a running Grafana
instance via its HTTP API. Re-run anytime after editing a dashboard file to
re-sync. Dashboards are grouped under a single "CloudSentinel-XAI" folder.

Run:  python grafana/deploy_dashboards.py
Env:  GRAFANA_URL (default http://127.0.0.1:3000)
      GRAFANA_USER / GRAFANA_PASSWORD (default admin/admin)
"""
import glob
import json
import os

import requests

GRAFANA_URL = os.environ.get("GRAFANA_URL", "http://127.0.0.1:3000")
AUTH = (
    os.environ.get("GRAFANA_USER", "admin"),
    os.environ.get("GRAFANA_PASSWORD", "admin"),
)
FOLDER_TITLE = "CloudSentinel-XAI"
DASHBOARDS_DIR = os.path.join(os.path.dirname(__file__), "dashboards")


def ensure_folder() -> str:
    resp = requests.get(f"{GRAFANA_URL}/api/folders", auth=AUTH)
    resp.raise_for_status()
    for folder in resp.json():
        if folder["title"] == FOLDER_TITLE:
            return folder["uid"]
    resp = requests.post(f"{GRAFANA_URL}/api/folders", auth=AUTH, json={"title": FOLDER_TITLE})
    resp.raise_for_status()
    return resp.json()["uid"]


def get_datasource_uids() -> dict:
    """Look up this Grafana instance's REAL datasource UIDs by type. UIDs are
    auto-generated per-install, so the ones baked into the dashboard JSON
    files (from whoever last exported them) will not match on a different
    machine — that mismatch is what causes every panel to silently show
    "No data" with no obvious error."""
    resp = requests.get(f"{GRAFANA_URL}/api/datasources", auth=AUTH)
    resp.raise_for_status()
    uids = {}
    for ds in resp.json():
        uids.setdefault(ds["type"], ds["uid"])
    missing = {"prometheus", "loki"} - uids.keys()
    if missing:
        raise RuntimeError(
            f"This Grafana instance has no configured datasource(s) of type: {missing}. "
            "Add a Prometheus and a Loki datasource first (Connections > Data sources)."
        )
    return uids


def remap_datasource_uids(obj, uid_by_type: dict) -> None:
    """Recursively rewrite every {"type": ..., "uid": ...} datasource
    reference in-place to point at THIS instance's real UID, so the same
    dashboard JSON works no matter whose Grafana it's deployed to."""
    if isinstance(obj, dict):
        if "type" in obj and "uid" in obj and obj["type"] in uid_by_type:
            obj["uid"] = uid_by_type[obj["type"]]
        for v in obj.values():
            remap_datasource_uids(v, uid_by_type)
    elif isinstance(obj, list):
        for v in obj:
            remap_datasource_uids(v, uid_by_type)


def deploy() -> None:
    folder_uid = ensure_folder()
    uid_by_type = get_datasource_uids()
    for path in sorted(glob.glob(os.path.join(DASHBOARDS_DIR, "*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            dashboard = json.load(f)
        remap_datasource_uids(dashboard, uid_by_type)
        payload = {
            "dashboard": dashboard,
            "folderUid": folder_uid,
            "overwrite": True,
            "message": "Synced from repo via deploy_dashboards.py",
        }
        resp = requests.post(f"{GRAFANA_URL}/api/dashboards/db", auth=AUTH, json=payload)
        name = os.path.basename(path)
        print(f"OK   {name} -> {resp.json().get('url')}" if resp.ok
              else f"FAIL {name}: {resp.status_code} {resp.text}")


if __name__ == "__main__":
    deploy()
