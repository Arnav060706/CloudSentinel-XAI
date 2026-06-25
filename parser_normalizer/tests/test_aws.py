from src.pipeline import ParserPipeline

def test_successful_aws_parsing():
    pipeline = ParserPipeline()
    raw_log = {
        "eventTime": "2026-06-10T08:14:32Z",
        "eventType": "AwsConsoleSignIn",
        "userIdentity": {"arn": "arn:aws:iam::112233445566:user/alice.chen"},
        "sourceIPAddress": "203.0.113.45",
        "eventName": "ConsoleLogin",
        "responseElements": {"ConsoleLogin": "Success"}
    }
    
    result = pipeline.process_log("AWS", raw_log)
    
    assert result is not None
    assert result.source_cloud == "AWS"
    assert result.user_id == "arn:aws:iam::112233445566:user/alice.chen"
    assert result.status == "SUCCESS"

def test_aws_failed_validation():
    pipeline = ParserPipeline()
    raw_log = {
        "eventType": "AwsConsoleSignIn",
        "sourceIPAddress": "not-an-ip" # Invalid IP triggers Pydantic failure
    }
    result = pipeline.process_log("AWS", raw_log)
    
    assert result is None
    assert len(pipeline.failed_logs_store) == 1