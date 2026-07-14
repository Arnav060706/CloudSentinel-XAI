"""
enrichment.py

Adds three signals to the unified log record that are easy to get wrong
in a way that leaks the label into the features:

  - is_known_proxy_or_tor : a REAL IP reputation lookup (Tor exit-node
    list + hosting/ASN check), never derived from anything in the event
    that a human labeler could have used as a shortcut (e.g. tool name).
  - principal_created_in_window : whether this principal was created
    (CreateUser / "Add service principal" / CreateServiceAccount, etc.)
    within PRINCIPAL_CREATED_WINDOW_MINUTES of this event, tracked from
    genuine first-seen timestamps — never string-matched off the
    principal's name.
  - ua_family / ua_version : parsed from the user agent string.

Design note on streaming vs. batch:
This pipeline processes events one at a time, in arrival order, via
ParserPipeline.process_log(). PrincipalWindowTracker therefore only knows
about creation events it has already seen when it evaluates the current
event — same constraint any real-time detector has. An offline/batch job can scan the whole dataset
first and get the *true* earliest creation event regardless of order.
If you rebuild training data offline, prefer a batch first-seen pass
(see build_principal_first_seen-style logic) and reserve this streaming
version for live scoring.
"""

import re
import datetime
from typing import Optional, Tuple

try:
    import maxminddb
except ImportError:
    maxminddb = None

PRINCIPAL_CREATED_WINDOW_MINUTES = 30

KNOWN_HOSTING_ASN_KEYWORDS = [
    "digitalocean", "ovh", "hetzner", "amazon", "google",
    "microsoft", "vultr", "linode",
]  # tune per your threat model; cloud-provider-owned IPs warrant a
   # separate category from residential/VPN, not a blanket "malicious" flag

CREATION_ACTIONS = {
    "aws": {"CreateUser", "CreateRole", "CreateAccessKey"},
    "azure": {"Add user", "Add service principal", "Add application"},
    "gcp": {"google.iam.admin.v1.CreateServiceAccount"},
}

UA_PATTERNS = [
    (r"aws-cli/([\d.]+)", "aws-cli"),
    (r"aws-sdk-go/([\d.]+)", "aws-sdk-go"),
    (r"Boto3/([\d.]+)", "Boto3"),
    (r"azure-cli/([\d.]+)", "azure-cli"),
    (r"Terraform/([\d.]+)", "Terraform"),
    (r"google-cloud-sdk/([\d.]+)", "google-cloud-sdk"),
    (r"Chrome/([\d.]+)", "Chrome"),
    (r"Firefox/([\d.]+)", "Firefox"),
    (r"Safari/([\d.]+)", "Safari"),
    # Previously missing -> these all fell through to the ("Other", "Unknown")
    # fallback below, which graph_engine.py's fuzzy-fusion now deliberately
    # excludes from earning similarity credit (matching on "we don't know" is
    # not evidence of shared identity). Without real patterns for these, that
    # correct fix also threw away genuinely-informative UA signal for most of
    # this dataset's attack traffic, which predominantly uses exactly these
    # tools (see dataset_script/env_profile.py's SCRIPT_UAS/SUSPICIOUS_UAS).
    (r"curl/([\d.]+)", "curl"),
    (r"python-requests/([\d.]+)", "python-requests"),
    (r"python-urllib3/([\d.]+)", "python-urllib3"),
    (r"Go-http-client/([\d.]+)", "Go-http-client"),
]


# ---------------------------------------------------------------------------
# Reference data loading (call once, e.g. in ParserPipeline.__init__)
# ---------------------------------------------------------------------------

def load_tor_exit_list(path: str) -> Optional[set]:
    try:
        with open(path) as f:
            return set(line.strip() for line in f if line.strip() and not line.startswith("#"))
    except FileNotFoundError:
        return None


def load_geo_db(path: Optional[str]):
    if maxminddb is None or path is None:
        return None
    try:
        return maxminddb.open_database(path)
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------------------
# Per-event enrichment
# ---------------------------------------------------------------------------

def ip_reputation(ip: Optional[str], tor_set, asn_db):
    """Returns True/False, or the string 'Unknown' when no reference data is loaded."""
    if not ip:
        return "Unknown" if (tor_set is None and asn_db is None) else False
    if tor_set is not None and ip in tor_set:
        return True
    if asn_db is not None:
        try:
            result = asn_db.get(ip) or {}
            org = (result.get("autonomous_system_organization") or "").lower()
            if any(kw in org for kw in KNOWN_HOSTING_ASN_KEYWORDS):
                return True
        except Exception:
            pass
    if tor_set is None and asn_db is None:
        return "Unknown"
    return False


def geo_country(ip: Optional[str], country_db) -> str:
    if not ip or country_db is None:
        return "Unknown"
    try:
        result = country_db.get(ip) or {}
        return result.get("country", {}).get("iso_code", "Unknown")
    except Exception:
        return "Unknown"


def parse_user_agent(ua_string: Optional[str]) -> Tuple[str, str]:
    if not ua_string:
        return "Unknown", "Unknown"
    for pattern, family in UA_PATTERNS:
        m = re.search(pattern, ua_string)
        if m:
            return family, m.group(1)
    return "Other", "Unknown"


# ---------------------------------------------------------------------------
# Principal identity / creation-action extraction (raw-log level, adapted
# to this repo's actual raw schema — see mock_data/unified_datastream.json)
# ---------------------------------------------------------------------------

def _get_identity(cloud: str, raw_log: dict) -> dict:
    if cloud == "aws":
        return raw_log.get("userIdentity", {}) or {}
    if cloud == "azure":
        return raw_log.get("properties", {}) or {}
    if cloud == "gcp":
        return raw_log.get("protoPayload", {}).get("authenticationInfo", {}) or {}
    return {}


def _get_action(cloud: str, raw_log: dict):
    if cloud == "aws":
        return raw_log.get("eventName")
    if cloud == "azure":
        op = raw_log.get("operationName")
        return op.get("value") if isinstance(op, dict) else op
    if cloud == "gcp":
        return raw_log.get("protoPayload", {}).get("methodName")
    return None


def _get_timestamp(cloud: str, raw_log: dict) -> Optional[datetime.datetime]:
    field = {"aws": "eventTime", "azure": "time", "gcp": "timestamp"}.get(cloud)
    raw_ts = raw_log.get(field) if field else None
    if not raw_ts:
        return None
    try:
        return datetime.datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def principal_key(cloud: str, identity: dict) -> Optional[str]:
    if cloud == "aws":
        return identity.get("arn") or identity.get("userName") or identity.get("principalId")
    if cloud == "azure":
        return identity.get("userPrincipalName") or identity.get("appId") or identity.get("userId")
    if cloud == "gcp":
        return identity.get("principalEmail") or identity.get("principalSubject")
    return None


def principal_type_from_identity(cloud: str, identity: dict) -> str:
    if cloud == "aws":
        return identity.get("type", "Unknown")  # IAMUser, AssumedRole, Root, FederatedUser, AWSService
    if cloud == "azure":
        return "ServicePrincipal" if (identity.get("appId") and not identity.get("userPrincipalName")) else "User"
    if cloud == "gcp":
        email = identity.get("principalEmail", "") or ""
        return "ServiceAccount" if email.endswith(".iam.gserviceaccount.com") else "User"
    return "Unknown"


class PrincipalWindowTracker:
    """
    Stateful, streaming approximation of "was this principal created
    recently?". Holds first-seen creation timestamps per principal and
    is meant to live for the lifetime of a ParserPipeline instance.
    """

    def __init__(self, window_minutes: int = PRINCIPAL_CREATED_WINDOW_MINUTES):
        self.window_minutes = window_minutes
        self._first_seen = {}

    def observe_and_check(self, cloud: str, raw_log: dict) -> bool:
        cloud = cloud.lower()
        identity = _get_identity(cloud, raw_log)
        key = principal_key(cloud, identity)
        ts = _get_timestamp(cloud, raw_log)
        action = _get_action(cloud, raw_log)

        if key and ts and action in CREATION_ACTIONS.get(cloud, set()):
            existing = self._first_seen.get(key)
            if existing is None or ts < existing:
                self._first_seen[key] = ts

        if not key or not ts or key not in self._first_seen:
            return False

        delta_minutes = (ts - self._first_seen[key]).total_seconds() / 60.0
        return 0 <= delta_minutes <= self.window_minutes
