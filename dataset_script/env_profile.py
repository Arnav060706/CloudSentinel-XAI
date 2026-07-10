"""
env_profile.py — Procedurally-generated environment, calibrated to real logs
============================================================================

The company's real AWS/Azure/GCP logs are used as a *format and distribution
reference*, not a hard boundary. This module PROCEDURALLY GENERATES a large,
seeded, realistic organization (hundreds of identities, hundreds of IPs, many
devices/geos) whose shapes match the real logs, so a model can learn BEHAVIOUR
rather than memorize a tiny vocabulary.

CRITICAL DESIGN PROPERTY — label neutrality (anti-leakage):
    No identity is inherently "good" or "bad". Every human/service identity is
    role-typed but label-neutral; attack scenarios COMPROMISE randomly-drawn
    LEGITIMATE identities. The only attack-exclusive identities are the ones an
    attacker *creates mid-attack* (backdoor users/keys) — which is realistic,
    and whose creation event is itself the signal. This prevents the model from
    learning "username X => malicious".

Anchored to the real logs: the real users/IPs/UAs are always INCLUDED in the
generated population (so validate_realism.py still shows high overlap), then
the population is expanded around them.
"""
from __future__ import annotations
import random, hashlib

TENANT_ID = "aaaabbbb-1111-2222-3333-ccccddddeeee"
CORP_DOMAIN = "corp-example.com"
AWS_ACCOUNTS = ["112233445566", "998877665544"]
GCP_PROJECT = "proj-alpha-112233"
AWS_REGIONS = ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1", "ap-northeast-1"]

# --- Seeds anchored to the REAL logs (always present in the population) ----
_REAL_HUMAN_USERS = [
    "alice.chen", "bob.smith", "carol.jones", "dan.wu", "emma.frost",
    "frank.miller", "grace.lee", "henry.park", "irene.zhao", "james.nguyen",
    "kate.brown", "liam.foster", "mia.white",
]
_REAL_SERVICE = ["devops-svc", "svc-ci-pipeline", "svc-cicd"]
_REAL_CORP_IPS = ["203.0.113.45", "203.0.113.10", "203.0.113.30", "203.0.113.60",
                  "203.0.113.71", "203.0.113.88", "203.0.113.90"]
_REAL_INTERNAL_IPS = ["10.0.1.50", "10.0.2.15", "10.0.0.25", "10.128.0.5"]
_REAL_HOME_IPS = ["198.51.100.22", "198.51.100.55", "198.51.100.88"]
_REAL_TOR = ["185.220.101.47"]
_REAL_HOSTING = ["91.108.4.180", "91.108.56.180", "103.21.244.0"]
_REAL_FOREIGN = ["220.100.52.33", "203.0.114.200"]

BROWSER_UAS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Mozilla/5.0 (X11; Linux x86_64)",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0)",
]
SDK_UAS_AWS = ["Boto3/1.26.0", "Boto3/1.26.0 Python/3.10.0", "aws-cli/2.13.0", "aws-cli/2.15.0"]
SDK_UAS_GCP = ["google-cloud-sdk/471.0.0", "google-cloud-sdk/471.0.0 command/gcloud",
               "google-cloud-sdk/463.0.0"]
SCRIPT_UAS = ["python-requests/2.28.0", "python-requests/2.31.0", "curl/7.88.1"]
SUSPICIOUS_UAS = ["python-requests/2.31.0", "curl/7.88.1", "Go-http-client/1.1",
                  "python-urllib3/1.26"]

BENIGN_GEO = [("San Jose", "California", "US"), ("Seattle", "Washington", "US"),
              ("Austin", "Texas", "US"), ("New York", "New York", "US"),
              ("Denver", "Colorado", "US")]
FOREIGN_GEO = [("Moscow", "Moscow", "RU"), ("Kyiv", "Kyiv", "UA"),
               ("Shanghai", "Shanghai", "CN"), ("Berlin", "Berlin", "DE"),
               ("Lagos", "Lagos", "NG"), ("Tehran", "Tehran", "IR")]

_FIRST = ["alex","sam","jordan","taylor","morgan","chris","pat","robin","jamie","casey",
          "drew","lee","noah","ava","liam","emma","olivia","ethan","sophia","mason",
          "isabella","logan","mila","lucas","aria","jack","ella","ryan","nora","owen",
          "priya","raj","wei","yuki","omar","fatima","ivan","elena","diego","sofia"]
_LAST = ["patel","kim","garcia","nguyen","muller","rossi","silva","khan","cohen","wang",
         "ali","reyes","brooks","hayes","ward","price","bennett","hughes","ross","cole",
         "sharma","tanaka","novak","haas","costa","dubois","meyer","larsen","ivanov","park"]


class Environment:
    """A generated organization. Deterministic given the seed."""

    def __init__(self, seed: int = 42, n_users: int = 300, n_services: int = 25,
                 n_corp_ips: int = 250, n_home_ips: int = 120,
                 n_tor: int = 40, n_hosting: int = 60, n_foreign: int = 80,
                 holdout_frac: float = 0.15):
        self.rng = random.Random(seed)
        self.seed = seed

        # --- Identities (role-typed, LABEL-NEUTRAL) ---
        # roles: human_admin (~8%), human_user (~82%), service_account
        self.users = {}          # name -> role
        for u in _REAL_HUMAN_USERS:
            self.users[u] = "human_admin" if u in ("alice.chen","carol.jones","grace.lee") else "human_user"
        seen = set(self.users)
        while len(self.users) < n_users:
            name = f"{self.rng.choice(_FIRST)}.{self.rng.choice(_LAST)}"
            if name in seen: 
                name = f"{name}{self.rng.randint(1,99)}"
            seen.add(name)
            role = self.rng.choices(["human_admin","human_user"], weights=[0.08,0.92])[0]
            self.users[name] = role

        self.services = {}
        for s in _REAL_SERVICE:
            self.services[s] = "service_account"
        i = 0
        svc_kinds = ["ci","cd","backup","deploy","monitor","etl","scanner","terraform","k8s"]
        while len(self.services) < n_services:
            s = f"svc-{self.rng.choice(svc_kinds)}-{i:02d}"; i += 1
            self.services[s] = "service_account"

        self.human_names = [u for u,r in self.users.items()]
        self.admin_names = [u for u,r in self.users.items() if r == "human_admin"]
        self.service_names = list(self.services)

        # Home cloud per user (stable) so most users are naturally single-cloud
        self.home_cloud = {u: ["aws","azure","gcp"][int(hashlib.md5(u.encode()).hexdigest(),16) % 3]
                           for u in self.human_names}

        # --- Held-out identities: never appear in benign/train split, used to
        #     test generalization to UNSEEN principals. ---
        pool = list(self.human_names)
        self.rng.shuffle(pool)
        n_hold = int(len(pool) * holdout_frac)
        self.holdout_users = set(pool[:n_hold])
        self.train_users = [u for u in self.human_names if u not in self.holdout_users]

        # --- IP pools (real ones always included, then expanded) ---
        self.corp_ips = self._expand(_REAL_CORP_IPS, "203.0.113.", n_corp_ips)
        self.internal_ips = self._expand(_REAL_INTERNAL_IPS, "10.0.", n_corp_ips//3, internal=True)
        self.home_ips = self._expand(_REAL_HOME_IPS, "198.51.100.", n_home_ips)
        self.tor_ips = self._expand(_REAL_TOR, "185.220.101.", n_tor)
        self.hosting_ips = self._expand(_REAL_HOSTING, "91.108.", n_hosting, two_octet=True)
        self.foreign_ips = self._expand(_REAL_FOREIGN, "220.100.", n_foreign, two_octet=True)

        self.benign_ips = self.corp_ips + self.internal_ips + self.home_ips
        self.malicious_ips = self.tor_ips + self.hosting_ips + self.foreign_ips

    def _expand(self, real, prefix, n, internal=False, two_octet=False):
        out = list(real); seen = set(real)
        while len(out) < max(n, len(real)):
            if two_octet:
                ip = f"{prefix}{self.rng.randint(0,255)}.{self.rng.randint(1,254)}"
            elif internal:
                ip = f"10.0.{self.rng.randint(0,5)}.{self.rng.randint(1,254)}"
            else:
                ip = f"{prefix}{self.rng.randint(1,254)}"
            if ip not in seen: seen.add(ip); out.append(ip)
        return out

    # --- convenience pickers ---
    def upn(self, user): return f"{user}@{CORP_DOMAIN}"
    def gcp_principal(self, user, is_service):
        return (f"{user}@{GCP_PROJECT}.iam.gserviceaccount.com" if is_service
                else self.upn(user))
    def pick_benign_ip(self): return self.rng.choice(self.benign_ips)
    def pick_ip(self, infra):
        pools = {"office": self.corp_ips, "home": self.home_ips,
                 "internal": self.internal_ips, "tor": self.tor_ips,
                 "hosting": self.hosting_ips, "foreign": self.foreign_ips}
        return self.rng.choice(pools.get(infra, self.foreign_ips))


def ml_label(anomaly: bool, category: str, severity: float):
    """Exact ml_labels block shape used in the company logs (ground truth)."""
    return {"anomaly_flag": bool(anomaly),
            "threat_category": category,
            "severity_score": round(max(0.0, min(1.0, severity)), 2)}