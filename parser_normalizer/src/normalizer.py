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

def normalize_severity_from_score(score: float) -> str:
    """Maps a numeric severity score to a standard label."""
    # Ensure score is treated as a float
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = 0.0
        
    if score >= 4:
        return "CRITICAL"
    elif score >= 3:
        return "HIGH"
    elif score >= 2:
        return "MEDIUM"
    else:
        return "LOW"
