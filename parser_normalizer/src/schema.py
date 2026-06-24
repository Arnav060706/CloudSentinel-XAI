#Parsed data is passed to this schema.py file that defines the structure and validation rules for the data. It ensures that the data conforms to the expected format and types before further processing.
# The schema may include fields, data types, constraints, and relationships between different parts of the data.
# Here Pydantic handles the heavy lifting of ensuring required fields exist and IPs are valid.

from pydantic import BaseModel, IPvAnyAddress, Field
from typing import Optional, Any, List, Dict
from datetime import datetime

class UnifiedLogModel(BaseModel):
    timestamp: datetime
    source_cloud: str
    event_type: str
    user_id: str
    source_ip: Optional[IPvAnyAddress] = None
    destination_ip: Optional[IPvAnyAddress] = None
    resource: str
    action: str
    status: str
    severity: str
    raw_log: dict[str, Any]

    class Config:
        # Pydantic V2 config to ensure datetime is output as ISO 8601 strings in JSON (Standardizing datetime serialization for JSON output)
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
