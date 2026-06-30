# Parsed data is passed to this schema.py file that defines the structure and validation rules for the data. 
# It ensures that the data conforms to the expected format and types before further processing.
# Here Pydantic handles the heavy lifting of ensuring required fields exist and IPs are valid.

from pydantic import BaseModel, IPvAnyAddress, Field
from typing import Optional, Any, List, Dict
from datetime import datetime

class UnifiedLogModel(BaseModel):
    # --- YOUR EXISTING CORE FIELDS ---
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

    # --- NEW PHASE 1 & 2 TELEMETRY FIELDS (For ML Feature Extraction) ---
    mfa_authenticated: bool = False
    device_compliant_status: Optional[str] = None
    user_agent: Optional[str] = None
    geo_country: Optional[str] = None

    account_type: str = "UNKNOWN"  # "USER" or "SERVICE"
