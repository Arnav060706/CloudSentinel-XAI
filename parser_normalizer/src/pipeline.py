import logging
from pydantic import ValidationError
from src.schema import UnifiedLogModel
from src.parsers.aws_parser import AWSCloudTrailParser
from src.parsers.azure_parser import AzureActivityParser
from src.parsers.gcp_parser import GCPCloudAuditParser
from src.metrics import LOGS_RECEIVED, LOGS_NORMALIZED, LOGS_FAILED, PROCESS_TIME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cloud-specific signatures used for automatic source detection.
# Created once when the module is imported.
CLOUD_SIGNATURES = {
    "AWS": (
        "eventVersion",
        "userIdentity",
        "awsRegion",
        "eventID",
    ),
    "AZURE": (
        "callerIpAddress",
        "category",
        "correlationId",
        "tenantId",
    ),
    "GCP": (
        "protoPayload",
        "insertId",
        "logName",
        "receiveTimestamp",
    ),
}


class ParserPipeline:
    def __init__(self):
        self.parsers = {
            "AWS": AWSCloudTrailParser(),
            "AZURE": AzureActivityParser(),
            "GCP": GCPCloudAuditParser(),
        }
        self.failed_logs_store = []

    def _detect_cloud_source(self, raw_log: dict) -> str:
        """
        Detect the originating cloud provider by matching provider-specific
        signature keys against the incoming raw log.
        """

        scores = {
            cloud: sum(1 for key in keys if key in raw_log)
            for cloud, keys in CLOUD_SIGNATURES.items()
        }

        best_score = max(scores.values())

        # No matching signature found
        if best_score == 0:
            return "UNKNOWN"

        # Ambiguous match between multiple providers
        if list(scores.values()).count(best_score) > 1:
            return "UNKNOWN"

        return max(scores, key=scores.get)

    @PROCESS_TIME.time()
    def process_log(self,raw_log: dict,source_cloud: str = None,) -> UnifiedLogModel | None:

        LOGS_RECEIVED.inc()

        if not source_cloud:
            source_cloud = self._detect_cloud_source(raw_log)

        parser = self.parsers.get(source_cloud.upper())

        if not parser:
            error_msg = f"No parser configured or detected for cloud: {source_cloud}"
            logger.error(error_msg)
            self.failed_logs_store.append(
                {"error": error_msg, "raw_log": raw_log}
            )
            LOGS_FAILED.inc()
            return None

        try:
            parsed_dict = parser.parse(raw_log)

            validated_log = UnifiedLogModel(**parsed_dict)

            LOGS_NORMALIZED.inc()

            return validated_log

        except ValidationError as e:
            logger.error(f"Validation failed for log: {e}")
            self.failed_logs_store.append(
                {"error": str(e), "raw_log": raw_log}
            )
            LOGS_FAILED.inc()
            return None

        except Exception as e:
            logger.error(f"Processing error: {e}")
            self.failed_logs_store.append(
                {"error": str(e), "raw_log": raw_log}
            )
            LOGS_FAILED.inc()
            return None