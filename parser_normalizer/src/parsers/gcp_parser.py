from typing import Any, Dict

class GCPCloudAuditParser:
    def parse(self, raw_log: Dict[str, Any]) -> Dict[str, Any]:
        proto_payload = raw_log.get("protoPayload", {})
        
        # Determine Status
        status = "SUCCESS"
        if "status" in proto_payload and proto_payload["status"].get("code", 0) != 0:
            status = "FAILED"
            
        metadata = proto_payload.get("metadata", {})
        events = metadata.get("event", [])
        if events and isinstance(events, list):
            event_name = events[0].get("eventName", "").lower()
            if "failure" in event_name:
                status = "FAILED"

        # Determine Severity based on ML labels
        ml_labels = raw_log.get("ml_labels", {})
        severity_score = ml_labels.get("severity_score", 0.0)
        
        if severity_score >= 0.9:
            severity = "CRITICAL"
        elif severity_score >= 0.7:
            severity = "HIGH"
        elif severity_score >= 0.4:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        # --- NEW: Phase 1 & 2 Telemetry Extraction ---
        request_metadata = proto_payload.get("requestMetadata", {})
        user_agent = request_metadata.get("callerSuppliedUserAgent", "Unknown")
        mfa_authenticated = False
        geo_country = "Unknown"
        device_compliant_status = "Not Applicable"

        return {
            "timestamp": raw_log.get("timestamp"),
            "source_cloud": "GCP",
            "event_type": proto_payload.get("methodName", "Unknown"),
            "user_id": proto_payload.get("authenticationInfo", {}).get("principalEmail", "Unknown"),
            "source_ip": request_metadata.get("callerIp"),
            "destination_ip": None,
            "resource": proto_payload.get("resourceName", "Unknown"),
            "action": proto_payload.get("methodName", "Unknown"),
            "status": status,
            "severity": severity,
            "raw_log": raw_log,
            "mfa_authenticated": mfa_authenticated,
            "device_compliant_status": device_compliant_status,
            "user_agent": user_agent,
            "geo_country": geo_country
        }