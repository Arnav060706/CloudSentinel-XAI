"""
emitters.py — Produce log records in the EXACT AWS / Azure / GCP schemas used
in the company's real logs (CloudTrail, Azure SignIn/Audit, GCP Cloud Audit).

Each emitter returns one provider-native JSON record with a ground-truth
`ml_labels` block. Generators call these so both benign and attack data share
identical, schema-correct shapes that your ParserPipeline already understands.
"""
from __future__ import annotations
import random
from env_profile import (TENANT_ID, CORP_DOMAIN, GCP_PROJECT, AWS_REGIONS,
                         ml_label)

_counter = {"n": 0}
_run_tag = {"tag": ""}

def _eid(prefix):
    _counter["n"] += 1
    tag = f"-{_run_tag['tag']}" if _run_tag["tag"] else ""
    return f"{prefix}{tag}-{_counter['n']:05d}"

def reset_ids():
    _counter["n"] = 0

def set_run_tag(tag):
    """Tag every subsequent event ID with a short source label (e.g. 'train',
    'holdout', 'atk-fast', 'atk-slow') so IDs stay globally unique once the
    four separately-generated sources are merged. Each generator script runs
    in its own process and would otherwise restart its ID counter from 1,
    causing collisions after the merge (e.g. two unrelated events both named
    evt-aws-00001)."""
    _run_tag["tag"] = tag

# --------------------------------------------------------------------- AWS
def aws_event(*, ts, event_name, user_name, account_id, ip, ua,
              region=None, success=True, mfa=True, principal_type="IAMUser",
              request_params=None, response=None, error_code=None,
              labels=None, read_only=False):
    rng = random
    region = region or rng.choice(AWS_REGIONS)
    arn = f"arn:aws:iam::{account_id}:user/{user_name}" if principal_type == "IAMUser" \
        else f"arn:aws:sts::{account_id}:assumed-role/{user_name}"
    rec = {
        "eventVersion": "1.08",
        "userIdentity": {"type": principal_type,
                         "principalId": f"AIDA{user_name[:8].upper()}",
                         "arn": arn, "accountId": account_id, "userName": user_name},
        "eventTime": ts, "eventSource": _aws_source(event_name),
        "eventName": event_name, "awsRegion": region, "sourceIPAddress": ip,
        "userAgent": ua, "requestParameters": request_params,
        "responseElements": response,
        "additionalEventData": {"MFAUsed": "Yes" if mfa else "No"},
        "eventID": _eid("evt-aws"), "eventType": "AwsApiCall",
        "readOnly": read_only, "managementEvent": True,
        "ml_labels": labels or ml_label(False, "Normal", 0.05),
    }
    if event_name == "ConsoleLogin":
        rec["eventType"] = "AwsConsoleSignIn"
        rec["responseElements"] = {"ConsoleLogin": "Success" if success else "Failure"}
    if not success and error_code:
        rec["errorCode"] = error_code
        rec["errorMessage"] = error_code
    return rec

def _aws_source(name):
    if name in ("ConsoleLogin",): return "signin.amazonaws.com"
    if name in ("AssumeRole", "GetSessionToken"): return "sts.amazonaws.com"
    return "iam.amazonaws.com"

# ------------------------------------------------------------------- Azure
def azure_event(*, ts, operation, upn, display_name, ip, ua, success=True,
                mfa=True, country=("San Jose","California","US"),
                compliant=True, managed=True, labels=None, category="AuditLogs"):
    err = 0 if success else 50126
    props = {
        "id": _eid("evt-az"), "createdDateTime": ts,
        "userDisplayName": display_name, "userPrincipalName": upn,
        "userId": f"user-guid-{abs(hash(upn))%9999:04d}",
        "appDisplayName": "Microsoft Azure",
        "ipAddress": ip, "clientAppUsed": "Browser", "userAgent": ua,
        "status": {"errorCode": err, "failureReason": None if success else "Invalid credentials"},
        "deviceDetail": {"operatingSystem": "Windows 10", "browser": "Edge 115.0",
                         "isCompliant": compliant, "isManaged": managed},
        "location": {"city": country[0], "state": country[1], "countryOrRegion": country[2]},
        "mfaDetail": ({"authMethod": "Phone Sign-in", "authDetail": "MFA completed"} if mfa else {}),
    }
    return {
        "time": ts, "category": "SignInLogs" if operation == "Sign-in activity" else category,
        "operationName": operation, "resultType": str(err),
        "resultDescription": "Success" if success else "Failure",
        "callerIpAddress": ip, "tenantId": TENANT_ID, "properties": props,
        "ml_labels": labels or ml_label(False, "Normal", 0.05),
    }

# --------------------------------------------------------------------- GCP
def gcp_event(*, ts, method, principal_email, ip, ua, success=True,
              resource_name=None, severity="NOTICE", labels=None,
              request=None):
    payload = {
        "@type": "type.googleapis.com/google.cloud.audit.AuditLog",
        "serviceName": _gcp_service(method), "methodName": method,
        "resourceName": resource_name or f"projects/{GCP_PROJECT}",
        "authenticationInfo": {"principalEmail": principal_email,
                               "principalSubject": f"user:{principal_email}"},
        "requestMetadata": {"callerIp": ip, "callerSuppliedUserAgent": ua},
        "authorizationInfo": [{"resource": f"projects/{GCP_PROJECT}",
                               "permission": _gcp_perm(method), "granted": success}],
    }
    if request: payload["request"] = request
    if not success:
        payload["status"] = {"code": 7, "message": "PERMISSION_DENIED"}
    return {
        "insertId": _eid("evt-gcp"),
        "logName": f"projects/{GCP_PROJECT}/logs/cloudaudit.googleapis.com%2Factivity",
        "timestamp": ts, "severity": severity,
        "resource": {"type": "project", "labels": {"project_id": GCP_PROJECT}},
        "protoPayload": payload,
        "ml_labels": labels or ml_label(False, "Normal", 0.1),
    }

def _gcp_service(m):
    if "login" in m.lower(): return "login.googleapis.com"
    if "SetIamPolicy" in m or "GetIamPolicy" in m: return "cloudresourcemanager.googleapis.com"
    return "iam.googleapis.com"

def _gcp_perm(m):
    if "SetIamPolicy" in m: return "resourcemanager.projects.setIamPolicy"
    if "CreateServiceAccount" in m: return "iam.serviceAccounts.create"
    if "CreateServiceAccountKey" in m: return "iam.serviceAccountKeys.create"
    return "iam.roles.get"