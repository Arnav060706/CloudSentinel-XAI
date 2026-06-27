#from src.parsers.base import BaseParser
from src.normalizer import normalize_timestamp
from typing import Any

class AzureActivityParser:
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
        # Azure SignInLogs use resultType "0" for success. AuditLogs use "Success".
        result_type = str(raw_log.get("resultType", ""))
        if result_type == "0" or result_type.lower() == "success":
            status = "SUCCESS"
        else:
            status = "FAILED"

        # 5. Extract User ID and Resource dynamically based on the log category
        properties = raw_log.get("properties", {})
        user_id = "Unknown"
        resource = "Unknown"

        if event_type == "SignInLogs":
            # For sign-ins, grab the user/service principal and the app they are accessing
            user_id = properties.get("userPrincipalName") or properties.get("servicePrincipalName", "Unknown")
            resource = properties.get("appDisplayName") or properties.get("appId", "Unknown")
            
        elif event_type == "AuditLogs":
            # For audit logs, initiatedBy contains the actor, and targetResources contains the affected entity
            initiated_by = properties.get("initiatedBy", {})
            user_info = initiated_by.get("user", {})
            app_info = initiated_by.get("app", {})
            
            user_id = user_info.get("userPrincipalName") or app_info.get("displayName", "Unknown")
            
            target_resources = properties.get("targetResources", [])
            if target_resources and isinstance(target_resources, list):
                resource = target_resources[0].get("displayName") or target_resources[0].get("id", "Unknown")

        # 7. Return the structured dictionary for Pydantic validation
        return {
            "timestamp": timestamp,
            "source_cloud": "AZURE",
            "event_type": event_type,
            "user_id": user_id,
            "source_ip": source_ip,
            "destination_ip": None, # Azure AD logs do not log destination IP in this context
            "resource": resource,
            "action": action,
            "status": status,
            "raw_log": raw_log
        }