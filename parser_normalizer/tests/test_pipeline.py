import pytest
from src.pipeline import ParserPipeline
from datetime import timezone

def test_successful_aws_parsing():
    pipeline = ParserPipeline()
    raw_log = {
      "eventTime": "2026-06-25T14:30:00Z",
      "eventType": "AwsApiCall",
      "userIdentity": {"arn": "arn:aws:iam::1234:user/test"},
      "sourceIPAddress": "10.0.0.5",
      "eventName": "DescribeInstances",
      "severity": "INFO"
    }
    
    result = pipeline.process_log("AWS", raw_log)
    
    assert result is not None
    assert result.source_cloud == "AWS"
    assert str(result.source_ip) == "10.0.0.5"
    assert result.severity == "LOW" # Normalized from INFO
    assert result.timestamp.tzinfo == timezone.utc

def test_failed_validation():
    pipeline = ParserPipeline()
    # Missing required fields and bad IP
    raw_log = {
        "eventTime": "2026-06-25T14:30:00Z",
        "sourceIPAddress": "not-an-ip" 
    }
    
    result = pipeline.process_log("AWS", raw_log)
    
    assert result is None
    assert len(pipeline.failed_logs_store) == 1
    assert "validation error" in pipeline.failed_logs_store[0]["error"].lower()