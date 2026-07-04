from typing import Any, Dict
from src.normalizer import normalize_severity_from_score, normalize_timestamp

class GCPCloudAuditParser:
    def parse(self, raw_log: Dict[str, Any]) -> Dict[str, Any]:
        proto = raw_log.get("protoPayload", {})
        
        # Determine Status
        status = "FAILED" if proto.get("status", {}).get("code", 0) != 0 else "SUCCESS"
        
        # Severity mapping using the central normalizer
        score = raw_log.get("ml_labels", {}).get("severity_score", 0.0)
        # GCP scores might need scaling to match the 1-4 range of normalize_severity_from_score
        scaled_score = score * 5 
        severity = normalize_severity_from_score(scaled_score)

        return {
            # Was previously passed straight through from raw_log["timestamp"]
            # with no parsing, unlike AWS/Azure. That meant a malformed GCP
            # timestamp surfaced as an opaque Pydantic error instead of a
            # clean validation failure, and there was no guaranteed UTC
            # normalization. Now consistent with the other two parsers.
            "timestamp": normalize_timestamp(raw_log.get("timestamp", "")),
            "source_cloud": "GCP",
            "event_type": proto.get("methodName", "Unknown"),
            "user_id": proto.get("authenticationInfo", {}).get("principalEmail", "Unknown"),
            "source_ip": proto.get("requestMetadata", {}).get("callerIp"),
            "resource": proto.get("resourceName", "Unknown"),
            "action": proto.get("methodName", "Unknown"),
            "status": status,
            "severity": severity,
            "raw_log": raw_log,
            "user_agent": proto.get("requestMetadata", {}).get("callerSuppliedUserAgent", "Unknown"),
        }