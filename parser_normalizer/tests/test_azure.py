import pytest
from datetime import timezone
from src.pipeline import ParserPipeline

def test_successful_azure_signin_parsing():
    pipeline = ParserPipeline()
    
    # Mock data based on your azure_iam_logs.json SignInLogs
    raw_log = {
        "time": "2026-06-10T08:20:14Z",
        "category": "SignInLogs",
        "operationName": "Sign-in activity",
        "resultType": "0",
        "callerIpAddress": "203.0.113.10",
        "properties": {
            "userPrincipalName": "alice.chen@corp-example.com",
            "appDisplayName": "Microsoft Azure"
        },
        "ml_labels": {
            "anomaly_flag": False,
            "severity_score": 0.05
        }
    }
    
    result = pipeline.process_log("AZURE", raw_log)
    
    assert result is not None
    assert result.source_cloud == "AZURE"
    assert result.event_type == "SignInLogs"
    assert result.user_id == "alice.chen@corp-example.com"
    assert result.resource == "Microsoft Azure"
    assert str(result.source_ip) == "203.0.113.10"
    assert result.status == "SUCCESS"
    assert result.severity == "LOW"
    assert result.timestamp.tzinfo == timezone.utc

def test_successful_azure_audit_parsing():
    pipeline = ParserPipeline()
    
    # Mock data based on your azure_iam_logs.json AuditLogs
    raw_log = {
        "time": "2026-06-10T10:15:00Z",
        "category": "AuditLogs",
        "operationName": "Add member to role",
        "resultType": "Success",
        "callerIpAddress": "185.220.101.47",
        "properties": {
            "initiatedBy": {
                "user": {
                    "userPrincipalName": "carol.jones@corp-example.com"
                }
            },
            "targetResources": [
                {
                    "displayName": "backdoor.user",
                    "type": "User"
                }
            ]
        },
        "ml_labels": {
            "anomaly_flag": True,
            "severity_score": 0.96
        }
    }
    
    result = pipeline.process_log("AZURE", raw_log)
    
    assert result is not None
    assert result.source_cloud == "AZURE"
    assert result.event_type == "AuditLogs"
    assert result.user_id == "carol.jones@corp-example.com"
    assert result.resource == "backdoor.user"
    assert str(result.source_ip) == "185.220.101.47"
    assert result.status == "SUCCESS"
    assert result.severity == "CRITICAL" # Elevated by ml_labels severity_score >= 0.8

def test_azure_failed_validation():
    pipeline = ParserPipeline()
    
    # Missing timestamp and invalid IP address format
    raw_log = {
        "category": "SignInLogs",
        "operationName": "Sign-in activity",
        "resultType": "0",
        "callerIpAddress": "not-a-valid-ip",
        "properties": {
            "userPrincipalName": "test@corp-example.com"
        }
    }
    
    result = pipeline.process_log("AZURE", raw_log)
    
    assert result is None
    assert len(pipeline.failed_logs_store) == 1
    
    error_message = pipeline.failed_logs_store[0]["error"].lower()
    assert "validation" in error_message or "timestamp" in error_message