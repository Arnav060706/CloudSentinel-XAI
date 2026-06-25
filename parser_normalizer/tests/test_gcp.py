from src.pipeline import ParserPipeline

def test_successful_gcp_parsing():
    pipeline = ParserPipeline()
    raw_log = {
        "timestamp": "2026-06-10T08:30:00Z",
        "protoPayload": {
            "methodName": "SetIamPolicy",
            "authenticationInfo": {"principalEmail": "alice.chen@corp-example.com"},
            "requestMetadata": {"callerIp": "203.0.113.45"},
            "resourceName": "projects/proj-alpha-112233"
        },
        "ml_labels": {"severity_score": 0.8} # Should flag as HIGH severity
    }
    
    result = pipeline.process_log("GCP", raw_log)
    
    assert result is not None
    assert result.source_cloud == "GCP"
    assert result.user_id == "alice.chen@corp-example.com"
    assert result.severity == "HIGH"
    assert result.status == "SUCCESS"

def test_gcp_error_status_parsing():
    pipeline = ParserPipeline()
    raw_log = {
        "timestamp": "2026-06-10T11:30:00Z",
        "protoPayload": {
            "methodName": "SetIamPolicy",
            "status": {"code": 7, "message": "The caller does not have permission"},
            "authenticationInfo": {"principalEmail": "ext-user@external-domain.com"},
            "requestMetadata": {"callerIp": "203.0.114.200"}
        }
    }
    
    result = pipeline.process_log("GCP", raw_log)
    assert result is not None
    assert result.status == "FAILED"