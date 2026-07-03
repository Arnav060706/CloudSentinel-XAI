from src.normalizer import normalize_timestamp, normalize_severity_from_score
from typing import Any

class AzureActivityParser:
    def parse(self, raw_log: dict[str, Any]) -> dict[str, Any]:
        properties = raw_log.get("properties", {})
        
        # Status mapping
        result_type = str(raw_log.get("resultType", ""))
        status = "FAILED" if (result_type != "0" and result_type.lower() != "success") else "SUCCESS"
        
        # Extract ID and Resource
        user_id = properties.get("userPrincipalName") or properties.get("servicePrincipalName", "Unknown")
        target_resources = properties.get("targetResources", [])
        resource = target_resources[0].get("displayName") or target_resources[0].get("id", "Unknown") if target_resources else "Unknown"

        # Severity mapping using the central normalizer
        score = raw_log.get("ml_labels", {}).get("severity_score", 0.0)
        # Assuming score is 0-5 scale; adjust if your model outputs 0-1
        severity = normalize_severity_from_score(score)

        return {
            "timestamp": normalize_timestamp(raw_log.get("time", "")),
            "source_cloud": "AZURE",
            "event_type": raw_log.get("category", "Unknown"),
            "user_id": user_id,
            "source_ip": properties.get("ipAddress"),
            "resource": resource,
            "action": raw_log.get("operationName", {}).get("value") if isinstance(raw_log.get("operationName"), dict) else raw_log.get("operationName"),
            "status": status,
            "severity": severity,
            "raw_log": raw_log,
            "user_agent": properties.get("userAgent", "Unknown"),
        }