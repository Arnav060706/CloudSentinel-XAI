from typing import Any, Dict

class GCPCloudAuditParser:
    def parse(self, raw_log: Dict[str, Any]) -> Dict[str, Any]:
        proto_payload = raw_log.get("protoPayload", {})
        
        # Determine Status
        status = "SUCCESS"
        # 1. Check for standard API errors
        if "status" in proto_payload and proto_payload["status"].get("code", 0) != 0:
            status = "FAILED"
        # 2. Check for explicit Login failures
        metadata = proto_payload.get("metadata", {})
        events = metadata.get("event", [])
        if events and isinstance(events, list):
            event_name = events[0].get("eventName", "").lower()
            if "failure" in event_name:
                status = "FAILED"

        return {
            "timestamp": raw_log.get("timestamp"),
            "source_cloud": "GCP",
            "event_type": proto_payload.get("methodName", "Unknown"),
            "user_id": proto_payload.get("authenticationInfo", {}).get("principalEmail", "Unknown"),
            "source_ip": proto_payload.get("requestMetadata", {}).get("callerIp"),
            "destination_ip": None,
            "resource": proto_payload.get("resourceName", "Unknown"),
            "action": proto_payload.get("methodName", "Unknown"),
            "status": status,
            "raw_log": raw_log
        }