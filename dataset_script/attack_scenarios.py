"""
attack_scenarios.py — MITRE ATT&CK cross-cloud kill chains
==========================================================

Each scenario is an ordered list of steps. A step names the cloud, the API
action, a MITRE technique (id + name + tactic), the actor/principal, the
infrastructure to use (proxy/tor/foreign), and a severity band. The generator
turns these into schema-correct log records with GROUND-TRUTH labels derived
from the step (never from event content).

Coverage goes BEYOND the categories in the company's sample logs: alongside
BruteForce / PrivilegeEscalation / CredentialCreation / Reconnaissance /
AccountCompromised, we add Defense Evasion (MFA/logging tamper), Persistence
(login profile / SSH keys), Impact (resource deletion), and Exfiltration.
"""
from __future__ import annotations

# (tactic, technique_id, technique_name)
T = {
    "valid_accounts":   ("Initial Access",      "T1078",     "Valid Accounts"),
    "brute_force":      ("Credential Access",    "T1110",     "Brute Force"),
    "cloud_accounts":   ("Initial Access",      "T1078.004", "Valid Accounts: Cloud Accounts"),
    "create_account":   ("Persistence",         "T1136.003", "Create Account: Cloud Account"),
    "add_credentials":  ("Persistence",         "T1098.001", "Account Manipulation: Additional Cloud Credentials"),
    "add_roles":        ("Privilege Escalation", "T1098.003", "Account Manipulation: Additional Cloud Roles"),
    "modify_policy":    ("Privilege Escalation", "T1548",     "Abuse Elevation Control Mechanism"),
    "account_disc":     ("Discovery",           "T1087.004", "Account Discovery: Cloud Account"),
    "perm_groups_disc": ("Discovery",           "T1069.003", "Permission Groups Discovery: Cloud Groups"),
    "modify_auth":      ("Defense Evasion",     "T1556.006", "Modify Authentication Process: MFA"),
    "impair_logging":   ("Defense Evasion",     "T1562.008", "Impair Defenses: Disable Cloud Logs"),
    "unused_regions":   ("Defense Evasion",     "T1535",     "Unused/Unsupported Cloud Regions"),
    "login_profile":    ("Persistence",         "T1098.004", "Account Manipulation: SSH Auth Keys"),
    "assume_role":      ("Lateral Movement",    "T1550.001", "Use Alternate Auth Material: App Access Token"),
    "temp_creds":       ("Credential Access",    "T1552.005", "Cloud Instance Metadata / Temp Creds"),
    "exfil":            ("Exfiltration",        "T1537",     "Transfer Data to Cloud Account"),
    "resource_hijack":  ("Impact",              "T1496",     "Resource Hijacking"),
    "destroy":          ("Impact",              "T1485",     "Data Destruction"),
}

# infra tags: office | home | tor | hosting | foreign
def step(cloud, action, tech, sev, cat, infra="foreign", success=True,
         principal=None, note=""):
    tactic, tid, tname = T[tech]
    return {"cloud": cloud, "action": action, "tactic": tactic,
            "technique_id": tid, "technique_name": tname, "severity": sev,
            "category": cat, "infra": infra, "success": success,
            "principal": principal, "note": note}

# --------------------------------------------------------------------------
# SCENARIO 1 — Classic cross-cloud credential-theft APT (AWS -> Azure -> GCP)
# Mirrors + extends the pattern hinted at in the real logs.
# --------------------------------------------------------------------------
SCENARIO_APT_CROSS_CLOUD = {
    "name": "cross_cloud_apt_credential_theft",
    "victim_role": "human_admin",   # generator draws a random legit admin
    "compromised": True,
    "steps": [
        step("aws", "ConsoleLogin", "brute_force", 0.62, "BruteForce", "tor", success=False),
        step("aws", "ConsoleLogin", "brute_force", 0.78, "BruteForce", "tor", success=False),
        step("aws", "ConsoleLogin", "valid_accounts", 0.85, "AccountCompromised", "tor", success=True),
        step("aws", "ListUsers", "account_disc", 0.55, "Reconnaissance", "tor"),
        step("aws", "ListRoles", "perm_groups_disc", 0.55, "Reconnaissance", "tor"),
        step("aws", "CreateUser", "create_account", 0.80, "CredentialCreation", "tor",
             note="creates backdoor user bd-svc-01"),
        step("aws", "CreateAccessKey", "add_credentials", 0.83, "CredentialCreation", "tor"),
        step("aws", "AttachUserPolicy", "modify_policy", 0.89, "PrivilegeEscalation", "tor"),
        step("aws", "DeactivateMFADevice", "modify_auth", 0.91, "DefenseEvasion", "tor"),
        step("azure", "Sign-in activity", "cloud_accounts", 0.84, "SuspiciousLogin", "foreign"),
        step("azure", "Add member to role", "add_roles", 0.90, "PrivilegeEscalation", "foreign"),
        step("azure", "Update conditional access policy", "modify_auth", 0.88, "DefenseEvasion", "foreign"),
        step("gcp", "google.login.LoginService.loginSuccess", "cloud_accounts", 0.82, "SuspiciousLogin", "hosting"),
        step("gcp", "google.iam.admin.v1.CreateServiceAccount", "create_account", 0.86,
             "SuspiciousServiceAccountCreation", "hosting", note="exfil-agent svc"),
        step("gcp", "google.iam.admin.v1.CreateServiceAccountKey", "add_credentials", 0.88,
             "CredentialCreation", "hosting"),
        step("gcp", "SetIamPolicy", "modify_policy", 0.92, "PrivilegeEscalation", "hosting"),
        step("gcp", "SetIamPolicy", "exfil", 0.95, "UnauthorizedAccessAttempt", "hosting",
             note="grants external principal access to data bucket"),
    ],
}

# --------------------------------------------------------------------------
# SCENARIO 2 — Insider privilege abuse (legit account, no brute force)
# Harder case: valid creds, off-hours, slow, single->dual cloud.
# --------------------------------------------------------------------------
SCENARIO_INSIDER = {
    "name": "insider_privilege_abuse",
    "victim_role": "human_user",
    "compromised": False,   # genuine insider, not a stolen credential
    "steps": [
        step("aws", "ConsoleLogin", "valid_accounts", 0.35, "SuspiciousLogin", "home", success=True,
             note="off-hours but valid"),
        step("aws", "ListAttachedUserPolicies", "account_disc", 0.5, "Reconnaissance", "home"),
        step("aws", "PutUserPolicy", "modify_policy", 0.82, "PrivilegeEscalation", "home",
             note="self-grants inline admin policy"),
        step("aws", "GetSessionToken", "temp_creds", 0.7, "AccountCompromised", "home"),
        step("gcp", "GetIamPolicy", "account_disc", 0.5, "Reconnaissance", "home"),
        step("gcp", "SetIamPolicy", "modify_policy", 0.85, "PrivilegeEscalation", "home"),
    ],
}

# --------------------------------------------------------------------------
# SCENARIO 3 — Service-account key abuse + resource hijack (automation path)
# --------------------------------------------------------------------------
SCENARIO_SVC_ABUSE = {
    "name": "service_account_key_abuse",
    "victim_role": "service_account",
    "compromised": True,
    "service_account": True,
    "steps": [
        step("gcp", "google.iam.admin.v1.ListServiceAccounts", "account_disc", 0.6,
             "Reconnaissance", "hosting"),
        step("gcp", "google.iam.admin.v1.CreateServiceAccountKey", "add_credentials", 0.85,
             "CredentialCreation", "hosting"),
        step("aws", "AssumeRole", "assume_role", 0.8, "AccountCompromised", "hosting"),
        step("aws", "CreateServiceLinkedRole", "add_roles", 0.83, "PrivilegeEscalation", "hosting"),
        step("gcp", "SetIamPolicy", "resource_hijack", 0.9, "UnauthorizedAccessAttempt", "hosting",
             note="spins compute for cryptomining"),
    ],
}

# --------------------------------------------------------------------------
# SCENARIO 4 — Destructive attack: logging tamper then deletion (Impact)
# --------------------------------------------------------------------------
SCENARIO_DESTRUCTIVE = {
    "name": "logging_tamper_and_destruction",
    "victim_role": "human_user",
    "compromised": True,
    "steps": [
        step("aws", "ConsoleLogin", "valid_accounts", 0.8, "AccountCompromised", "hosting"),
        step("aws", "UpdateLoginProfile", "login_profile", 0.85, "PrivilegeEscalation", "hosting"),
        step("azure", "Disable account", "impair_logging", 0.9, "DefenseEvasion", "foreign",
             note="disables an admin's account to blind response"),
        step("aws", "DeleteRole", "destroy", 0.93, "UnauthorizedAccessAttempt", "hosting"),
        step("aws", "DeleteUser", "destroy", 0.94, "UnauthorizedAccessAttempt", "hosting"),
    ],
}

ALL_SCENARIOS = [SCENARIO_APT_CROSS_CLOUD, SCENARIO_INSIDER,
                 SCENARIO_SVC_ABUSE, SCENARIO_DESTRUCTIVE]