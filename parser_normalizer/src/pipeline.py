import logging
from pydantic import ValidationError
from src.schema import UnifiedLogModel
from src.parsers.aws_parser import AWSCloudTrailParser
from src.parsers.azure_parser import AzureActivityParser
from src.parsers.gcp_parser import GCPCloudAuditParser
from src.metrics import LOGS_RECEIVED, LOGS_NORMALIZED, LOGS_FAILED, PROCESS_TIME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParserPipeline:
    def __init__(self):
        self.parsers = {
            "AWS": AWSCloudTrailParser(),
            "AZURE": AzureActivityParser(),
            "GCP": GCPCloudAuditParser(),
        }
        self.failed_logs_store = [] 

    def _detect_cloud_source(self, raw_log: dict) -> str:
        """Dynamically inspects the raw JSON keys to identify the cloud provider."""
        # Check for GCP fingerprints
        if "protoPayload" in raw_log or "insertId" in raw_log:
            return "GCP"
        # Check for AWS fingerprints
        if "userIdentity" in raw_log or "eventVersion" in raw_log:
            return "AWS"
        # Check for Azure fingerprints
        if "callerIpAddress" in raw_log or "category" in raw_log:
            return "AZURE"
            
        return "UNKNOWN"

    @PROCESS_TIME.time()
    def process_log(self, raw_log: dict, source_cloud: str = None) -> UnifiedLogModel | None:
        LOGS_RECEIVED.inc()
        
        # Auto-detect the cloud if a tag wasn't explicitly provided
        if not source_cloud:
            source_cloud = self._detect_cloud_source(raw_log)
        
        parser = self.parsers.get(source_cloud.upper())
        if not parser:
            error_msg = f"No parser configured or detected for cloud: {source_cloud}"
            logger.error(error_msg)
            self.failed_logs_store.append({"error": error_msg, "raw_log": raw_log})
            LOGS_FAILED.inc()
            return None

        try:
            # 1. Parse and Normalize
            parsed_dict = parser.parse(raw_log)
            
            # 2. Validate using Pydantic
            validated_log = UnifiedLogModel(**parsed_dict)
            
            LOGS_NORMALIZED.inc()
            return validated_log

        except ValidationError as e:
            logger.error(f"Validation failed for log: {e}")
            self.failed_logs_store.append({"error": str(e), "raw_log": raw_log})
            LOGS_FAILED.inc()
            return None
        except Exception as e:
            logger.error(f"Processing error: {e}")
            self.failed_logs_store.append({"error": str(e), "raw_log": raw_log})
            LOGS_FAILED.inc()
            return None