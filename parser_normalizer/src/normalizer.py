from datetime import datetime, timezone
import dateutil.parser

def normalize_timestamp(ts_str: str) -> datetime:
    """Converts various timestamp formats to UTC timezone-aware datetime."""
    try:
        dt = dateutil.parser.parse(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception as e:
        raise ValueError(f"Invalid timestamp format: {ts_str}") from e

def normalize_severity(cloud: str, raw_severity: str) -> str:
    """Legacy string-based severity normalization."""
    raw_upper = str(raw_severity).upper()
    mapping = {
        "AWS": {"ERROR": "HIGH", "WARNING": "MEDIUM", "INFO": "LOW"},
        "AZURE": {"CRITICAL": "CRITICAL", "ERROR": "HIGH", "WARNING": "MEDIUM", "INFORMATIONAL": "LOW"}
    }
    cloud_map = mapping.get(cloud.upper(), {})
    return cloud_map.get(raw_upper, "UNKNOWN")

def normalize_severity_from_score(score: float) -> str:
    """
    Maps a numerical risk score (0.0 to 1.0) to a categorical severity.
    Useful when real-world logs include ML/Risk scores.
    """
    if score >= 0.85:
        return "CRITICAL"
    elif score >= 0.60:
        return "HIGH"
    elif score >= 0.30:
        return "MEDIUM"
    else:
        return "LOW"