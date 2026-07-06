from app.parser_normalizer.src.normalizer import normalize_timestamp, normalize_severity_from_score

class AWSCloudTrailParser:
    def parse(self, raw_log: dict) -> dict:
        user_identity = raw_log.get("userIdentity", {})
        identity_type = user_identity.get("type", "Unknown")

        # Classification logic
        if identity_type in {"IAMUser", "Root", "IdentityCenterUser", "FederatedUser", "SAMLUser"}:
            account_type = "USER"
        elif identity_type in {"AssumedRole", "Role", "AWSService"}:
            account_type = "SERVICE"
        else:
            account_type = "UNKNOWN"

        user_id = user_identity.get("arn", user_identity.get("principalId", identity_type))
        resource = raw_log.get("resources", [{}])[0].get("ARN", "None") if raw_log.get("resources") else "None"
        
        is_error = bool(raw_log.get("errorCode") or raw_log.get("errorMessage"))
        status = "FAILED" if is_error else "SUCCESS"

        score = raw_log.get("ml_labels", {}).get("severity_score")
        severity = normalize_severity_from_score(score) if score is not None else ("HIGH" if is_error else "LOW")

        # MFA signal was previously discarded even though CloudTrail carries
        # it on console logins (additionalEventData.MFAUsed). Losing this
        # made mfa_authenticated a constant False for every AWS event.
        additional_event_data = raw_log.get("additionalEventData", {}) or {}
        mfa_authenticated = str(additional_event_data.get("MFAUsed", "")).lower() == "yes"

        return {
            "timestamp": normalize_timestamp(raw_log.get("eventTime", "")),
            "source_cloud": "AWS",
            "event_type": raw_log.get("eventType", "Unknown"),
            "user_id": user_id,
            "source_ip": raw_log.get("sourceIPAddress"),
            "resource": resource,
            "action": raw_log.get("eventName"),
            "status": status,
            "severity": severity,
            "raw_log": raw_log,
            "account_type": account_type,
            "user_agent": raw_log.get("userAgent", "Unknown"),
            "mfa_authenticated": mfa_authenticated,
        }