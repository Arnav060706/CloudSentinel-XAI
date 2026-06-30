from src.parsers.base import BaseParser
from src.normalizer import normalize_timestamp, normalize_severity
from typing import Any

class AzureActivityParser(BaseParser):
    def parse(self, raw_log: dict[str, Any]) -> dict[str, Any]:
        # 1. Extract Timestamp
        time_str = raw_log.get("time", "")
        timestamp = normalize_timestamp(time_str)

        # 2. Extract Event Type and Action
        event_type = raw_log.get("category", "Unknown")
        action = raw_log.get("operationName", "Unknown")

        # 3. Extract Source IP
        source_ip = raw_log.get("callerIpAddress")

        # 4. Extract Status
        result_type = str(raw_log.get("resultType", ""))
        if result_type == "0" or result_type.lower() == "success":
            status = "SUCCESS"
        else:
            status = "FAILED"

        # 5. Extract User ID, Resource, and NEW Telemetry dynamically
        properties = raw_log.get("properties", {})
        user_id = "Unknown"
        resource = "Unknown"
        
        # New Telemetry Defaults
        mfa_authenticated = False
        device_compliant_status = "Unknown"
        geo_country = "Unknown"
        user_agent = properties.get("clientAppUsed", "Unknown")

        if event_type == "SignInLogs":
            user_id = properties.get("userPrincipalName") or properties.get("servicePrincipalName", "Unknown")
            resource = properties.get("appDisplayName") or properties.get("appId", "Unknown")
            
            # Extract advanced Azure telemetry
            device_detail = properties.get("deviceDetail", {})
            is_compliant = device_detail.get("isCompliant")
            if is_compliant is not None:
                device_compliant_status = str(is_compliant)
                
            location = properties.get("location", {})
            geo_country = location.get("countryOrRegion", "Unknown")
            
            auth_req = properties.get("authenticationRequirement")
            if auth_req and auth_req.lower() == "multifactorauthentication":
                mfa_authenticated = True
            
        elif event_type == "AuditLogs":
            initiated_by = properties.get("initiatedBy", {})
            user_info = initiated_by.get("user", {})
            app_info = initiated_by.get("app", {})
            
            user_id = user_info.get("userPrincipalName") or app_info.get("displayName", "Unknown")
            
            target_resources = properties.get("targetResources", [])
            if target_resources and isinstance(target_resources, list):
                resource = target_resources[0].get("displayName") or target_resources[0].get("id", "Unknown")

        # 6. Determine Severity
        raw_severity = "INFORMATIONAL"
        if status == "FAILED":
            raw_severity = "ERROR"
            
        ml_labels = raw_log.get("ml_labels", {})
        if ml_labels.get("anomaly_flag"):
            score = ml_labels.get("severity_score", 0.0)
            if score >= 0.8:
                raw_severity = "CRITICAL"
            elif score >= 0.5:
                raw_severity = "WARNING"

        # 7. Return the structured dictionary
        return {
            "timestamp": timestamp,
            "source_cloud": "AZURE",
            "event_type": event_type,
            "user_id": user_id,
            "source_ip": source_ip,
            "destination_ip": None,
            "resource": resource,
            "action": action,
            "status": status,
            "severity": normalize_severity("AZURE", raw_severity),
            "raw_log": raw_log,
            "mfa_authenticated": mfa_authenticated,
            "device_compliant_status": device_compliant_status,
            "user_agent": user_agent,
            "geo_country": geo_country
        }