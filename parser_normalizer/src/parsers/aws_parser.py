from src.parsers.base import BaseParser
from src.normalizer import normalize_timestamp, normalize_severity

class AWSCloudTrailParser(BaseParser):
    def parse(self, raw_log: dict) -> dict:
        return {
            "timestamp": normalize_timestamp(raw_log.get("eventTime", "")),
            "source_cloud": "AWS",
            "event_type": raw_log.get("eventType", "Unknown"),
            "user_id": raw_log.get("userIdentity", {}).get("arn", "Unknown"),
            "source_ip": raw_log.get("sourceIPAddress"),
            "destination_ip": None, # CloudTrail rarely logs dest IP
            "resource": raw_log.get("resources", [{"ARN": "None"}])[0].get("ARN"),
            "action": raw_log.get("eventName", "Unknown"),
            "status": "SUCCESS" if not raw_log.get("errorCode") else "FAILED",
            "severity": normalize_severity("AWS", raw_log.get("severity", "INFO")),
            "raw_log": raw_log
        }