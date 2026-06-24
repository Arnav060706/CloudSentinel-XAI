# This file is for normalization tasks
# Normalization is the process of transforming raw log data into a standardized format that can be easily analyzed and processed by downstream systems. 
# This may involve parsing, cleaning, and structuring the data to ensure consistency and compatibility across different sources and formats.

from datetime import datetime, timezone
import dateutil.parser

# Next function to normalize timestamps from various cloud providers to a standard UTC format. 
# This is essential for accurate time-based analysis and correlation of events across multi-cloud environments.

def normalize_timestamp(ts_str: str) -> datetime:
    """Converts various timestamp formats to UTC timezone-aware datetime."""
    try:
        dt = dateutil.parser.parse(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception as e:
        raise ValueError(f"Invalid timestamp format: {ts_str}") from e
    

# Next function to normalize the severity levels from different cloud providers to a unified scale (LOW, MEDIUM, HIGH, CRITICAL). 
# This is crucial for consistent risk assessment and alerting across multi-cloud environments.
    
def normalize_severity(cloud: str, raw_severity: str) -> str:
    """Maps provider-specific severities to a unified scale: LOW, MEDIUM, HIGH, CRITICAL"""
    raw_upper = raw_severity.upper()
    
    mapping = {
        "AWS": {"ERROR": "HIGH", "WARNING": "MEDIUM", "INFO": "LOW"},
        "AZURE": {"CRITICAL": "CRITICAL", "ERROR": "HIGH", "WARNING": "MEDIUM", "INFORMATIONAL": "LOW"}
    }
    
    cloud_map = mapping.get(cloud.upper(), {})
    return cloud_map.get(raw_upper, "UNKNOWN")