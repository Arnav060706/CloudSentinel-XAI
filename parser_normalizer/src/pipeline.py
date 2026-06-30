import logging
from pydantic import ValidationError

from src.schema import UnifiedLogModel
from src.parsers.aws_parser import AWSCloudTrailParser
from src.parsers.azure_parser import AzureActivityParser
from src.parsers.gcp_parser import GCPCloudAuditParser
from src.metrics import (
    LOGS_RECEIVED,
    LOGS_NORMALIZED,
    LOGS_FAILED,
    PROCESS_TIME,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cloud-specific signatures used for automatic source detection.
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
        Detect the cloud provider from provider-specific fields.
        """

        scores = {
            cloud: sum(1 for key in keys if key in raw_log)
            for cloud, keys in CLOUD_SIGNATURES.items()
        }

        best_score = max(scores.values())

        if best_score == 0:
            return "UNKNOWN"

        if list(scores.values()).count(best_score) > 1:
            return "UNKNOWN"

        return max(scores, key=scores.get)

    def classify_account(self, source_cloud: str, raw_log: dict) -> str:
        """
        Determine whether the identity is a USER or SERVICE account
        using the original cloud-native log.
        """

        try:
            # ---------------- AWS ----------------
            if source_cloud.upper() == "AWS":

                identity_type = (
                    raw_log.get("userIdentity", {})
                    .get("type", "")
                )

                if identity_type == "IAMUser":
                    return "USER"

                if identity_type in {
                    "AssumedRole",
                    "Role",
                    "AWSService",
                    "FederatedUser",
                }:
                    return "SERVICE"

            # ---------------- Azure ----------------
            elif source_cloud.upper() == "AZURE":

                props = raw_log.get("properties", {})

                if props.get("servicePrincipalId"):
                    return "SERVICE"

                if props.get("managedIdentity"):
                    return "SERVICE"

                if props.get("userPrincipalName"):
                    return "USER"

            # ---------------- GCP ----------------
            elif source_cloud.upper() == "GCP":

                email = (
                    raw_log.get("protoPayload", {})
                    .get("authenticationInfo", {})
                    .get("principalEmail", "")
                )

                if email.endswith("gserviceaccount.com"):
                    return "SERVICE"

                if email:
                    return "USER"

        except Exception as e:
            logger.warning(f"Account classification failed: {e}")

        return "UNKNOWN"

    @PROCESS_TIME.time()
    def process_log(
        self,
        raw_log: dict,
        source_cloud: str = None,
    ) -> UnifiedLogModel | None:

        LOGS_RECEIVED.inc()

        if not source_cloud:
            source_cloud = self._detect_cloud_source(raw_log)

        parser = self.parsers.get(source_cloud.upper())

        if not parser:
            error_msg = f"No parser configured for cloud: {source_cloud}"
            logger.error(error_msg)

            self.failed_logs_store.append(
                {
                    "error": error_msg,
                    "raw_log": raw_log,
                }
            )

            LOGS_FAILED.inc()
            return None

        try:

            parsed_dict = parser.parse(raw_log)

            # Classify account directly from the original raw log
            parsed_dict["account_type"] = self.classify_account(
                source_cloud,
                raw_log,
            )

            validated_log = UnifiedLogModel(**parsed_dict)

            LOGS_NORMALIZED.inc()

            return validated_log

        except ValidationError as e:

            logger.error(f"Validation failed: {e}")

            self.failed_logs_store.append(
                {
                    "error": str(e),
                    "raw_log": raw_log,
                }
            )

            LOGS_FAILED.inc()
            return None

        except Exception as e:

            logger.error(f"Processing error: {e}")

            self.failed_logs_store.append(
                {
                    "error": str(e),
                    "raw_log": raw_log,
                }
            )

            LOGS_FAILED.inc()
            return None