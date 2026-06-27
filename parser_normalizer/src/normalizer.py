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
