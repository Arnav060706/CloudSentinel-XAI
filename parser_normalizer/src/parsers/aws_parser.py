from src.parsers.base import BaseParser
from src.normalizer import normalize_timestamp, normalize_severity_from_score
import json

class AWSCloudTrailParser(BaseParser):
    def parse(self, raw_log: dict) -> dict:
        
        # --- 1. Defensive User Identity Extraction ---
        user_identity = raw_log.get("userIdentity", {})
        # Prioritize ARN, fallback to principalId, fallback to type
        user_id = user_identity.get("arn", user_identity.get("principalId", user_identity.get("type", "Unknown")))

        # --- 2. Defensive Resource Extraction ---
        resource = "None"
        if raw_log.get("resources") and len(raw_log["resources"]) > 0:
            resource = raw_log["resources"][0].get("ARN", "Unknown")
        elif raw_log.get("requestParameters"):
            # If no resources array, extract relevant data from request parameters
            req_params = raw_log["requestParameters"]
            if isinstance(req_params, dict):
                resource = req_params.get("roleArn", req_params.get("userName", "Multiple/Params"))

        # --- 3. Severity & Status Derivation ---
        is_error = bool(raw_log.get("errorCode") or raw_log.get("errorMessage"))
        status = "FAILED" if is_error else "SUCCESS"
        
        ml_labels = raw_log.get("ml_labels", {})
        severity_score = ml_labels.get("severity_score", None)
        
        if severity_score is not None:
            severity = normalize_severity_from_score(severity_score)
        else:
            severity = "HIGH" if is_error else "LOW"

        # --- 4. NEW: Phase 1 & 2 Telemetry Extraction ---
        # Parse MFA explicitly as a boolean
        mfa_str = str(user_identity.get("sessionContext", {}).get("attributes", {}).get("mfaAuthenticated")).lower()
        mfa_authenticated = (mfa_str == "true")
        
        user_agent = raw_log.get("userAgent", "Unknown")
        geo_country = "Unknown"
        device_compliant_status = "Not Applicable"

        # --- 5. Return matching the Unified Schema ---
        return {
            "timestamp": normalize_timestamp(raw_log.get("eventTime", "")),
            "source_cloud": "AWS",
            "event_type": raw_log.get("eventType", "Unknown"),
            "user_id": user_id,
            "source_ip": raw_log.get("sourceIPAddress"),
            "destination_ip": None,
            "resource": resource,
            "action": raw_log.get("eventName", "Unknown"),
            "status": status,
            "severity": severity,
            "raw_log": raw_log,
            "mfa_authenticated": mfa_authenticated,
            "device_compliant_status": device_compliant_status,
            "user_agent": user_agent,
            "geo_country": geo_country
        }