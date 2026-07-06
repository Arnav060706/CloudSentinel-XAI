import logging
import os
from pydantic import ValidationError
from app.parser_normalizer.src.schema import UnifiedLogModel
from app.parser_normalizer.src.parsers.aws_parser import AWSCloudTrailParser
from app.parser_normalizer.src.parsers.azure_parser import AzureActivityParser
from app.parser_normalizer.src.parsers.gcp_parser import GCPCloudAuditParser
from app.parser_normalizer.src.metrics import LOGS_RECEIVED, LOGS_NORMALIZED, LOGS_FAILED, PROCESS_TIME
from app.parser_normalizer.src import enrichment

# Optional reference data for IP reputation / geo lookups. Missing files are
# handled gracefully (fields fall back to "Unknown") - see enrichment.py.
REFERENCE_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reference_data")
TOR_LIST_PATH = os.path.join(REFERENCE_DATA_DIR, "tor_exit_nodes.txt")
GEO_COUNTRY_DB_PATH = os.path.join(REFERENCE_DATA_DIR, "GeoLite2-Country.mmdb")
GEO_ASN_DB_PATH = os.path.join(REFERENCE_DATA_DIR, "GeoLite2-ASN.mmdb")

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

        # Enrichment state: loaded once, reused across every event.
        self.tor_set = enrichment.load_tor_exit_list(TOR_LIST_PATH)
        self.country_db = enrichment.load_geo_db(GEO_COUNTRY_DB_PATH)
        self.asn_db = enrichment.load_geo_db(GEO_ASN_DB_PATH)
        self.principal_tracker = enrichment.PrincipalWindowTracker()

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

            # --- Enrichment (real IP reputation + real principal-creation
            #     tracking, ported from 06_normalize_features.py) ---
            cloud_key = source_cloud.lower()
            identity = enrichment._get_identity(cloud_key, raw_log)
            parsed_dict["principal_type"] = enrichment.principal_type_from_identity(cloud_key, identity)
            parsed_dict["principal_created_in_window"] = self.principal_tracker.observe_and_check(cloud_key, raw_log)
            parsed_dict["is_known_proxy_or_tor"] = enrichment.ip_reputation(
                parsed_dict.get("source_ip"), self.tor_set, self.asn_db
            )
            parsed_dict["geo_country"] = enrichment.geo_country(parsed_dict.get("source_ip"), self.country_db)
            ua_family, ua_version = enrichment.parse_user_agent(parsed_dict.get("user_agent"))
            parsed_dict["ua_family"], parsed_dict["ua_version"] = ua_family, ua_version

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