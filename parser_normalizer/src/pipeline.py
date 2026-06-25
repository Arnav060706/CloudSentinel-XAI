import logging
from pydantic import ValidationError
from src.schema import UnifiedLogModel
from src.parsers.aws_parser import AWSCloudTrailParser
from src.parsers.azure_parser import AzureActivityParser
from src.parsers.gcp_parser import GCPCloudAuditParser  # <-- Added GCP Import
from src.metrics import LOGS_RECEIVED, LOGS_NORMALIZED, LOGS_FAILED, PROCESS_TIME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParserPipeline:
    def __init__(self):
        # Router logic: map cloud source to parser instance
        self.parsers = {
            "AWS": AWSCloudTrailParser(),
            "AZURE": AzureActivityParser(),
            "GCP": GCPCloudAuditParser(),  # <-- Added GCP Router
        }
        self.failed_logs_store = [] 

    @PROCESS_TIME.time()
    def process_log(self, source_cloud: str, raw_log: dict) -> UnifiedLogModel | None:
        LOGS_RECEIVED.inc()
        
        parser = self.parsers.get(source_cloud.upper())
        if not parser:
            logger.error(f"No parser configured for cloud: {source_cloud}")
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