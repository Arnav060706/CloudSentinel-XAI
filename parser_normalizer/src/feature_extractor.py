import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging
import ipaddress
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)

class MLFeatureExtractor:
    def __init__(self):
        # Encoders for ALL categorical variables (Phase 1 & 2)
        self.label_encoders = {
            'source_cloud': LabelEncoder(),
            'action': LabelEncoder(),
            'user_id': LabelEncoder(),
            'device_compliant_status': LabelEncoder(),
            'browser_type': LabelEncoder(),
            'os_type': LabelEncoder(),
            'geo_country': LabelEncoder()
        }
        
        # Phase 1: Action Sensitivity Mapping (Basic Baseline)
        self.sensitivity_map = {
            "ConsoleLogin": 1, 
            "Sign-in activity": 1,
            "SetIamPolicy": 4,     # High sensitivity (Privilege changes)
            "DeleteBucket": 3,
            "CreateUser": 4,
            "DescribeInstances": 1 # Low sensitivity (Reconnaissance/Read-only)
        }
        
        # Target Variables (Must be isolated from X)
        self.target_columns = ['severity', 'severity_score', 'threat_category', 'anomaly_flag', 'trust_score', 'risk_score']

    def extract_features(
        self, 
        unified_logs: List[dict], 
        is_training: bool = True, 
        export_csv: bool = False,
        output_path_X: str = "features_X.csv",
        output_path_y: str = "targets_y.csv"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Converts a list of UnifiedLogModel dictionaries into an (X, y) tuple of Pandas DataFrames,
        strictly isolating target variables to prevent data leakage.
        Optionally exports the resulting matrices to CSV files.
        """
        if not unified_logs:
            return pd.DataFrame(), pd.DataFrame()

        # 1. Load logs and prepare the time index for stateful processing
        df = pd.DataFrame(unified_logs)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        # ==========================================
        # PHASE 1: BASELINE TELEMETRY
        # ==========================================
        
        # Temporal Context
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        df['is_night_access'] = df['hour_of_day'].apply(lambda x: 1 if x < 6 or x >= 22 else 0) # 10 PM to 6 AM
        df['is_business_hour'] = df['hour_of_day'].apply(lambda x: 1 if 8 <= x <= 18 else 0)

        # Identity & Authentication
        df['mfa_authenticated'] = df.get('mfa_authenticated', 0).fillna(0).astype(int)
        df['user_type_is_service'] = df['user_id'].astype(str).apply(
            lambda x: 1 if any(kw in x.lower() for kw in ['svc', 'role', 'arn:', 'service']) else 0
        )

        # Network & Endpoint (Browser/OS Parsed from user_agent)
        df['device_compliant_status'] = df.get('device_compliant_status', 'Unknown').fillna('Unknown')
        df['user_agent'] = df.get('user_agent', 'Unknown').fillna('Unknown')
        
        df['browser_type'] = df['user_agent'].apply(
            lambda x: "Chrome" if "Chrome" in str(x) else ("Firefox" if "Firefox" in str(x) else ("Safari" if "Safari" in str(x) else "Unknown"))
        )
        df['os_type'] = df['user_agent'].apply(
            lambda x: "Windows" if "Windows" in str(x) else ("Mac" if "Mac" in str(x) else ("Linux" if "Linux" in str(x) else "Unknown"))
        )

        # Action Baseline
        df['action_sensitivity_score'] = df['action'].map(self.sensitivity_map).fillna(1).astype(int)
        df['login_result_success'] = df['status'].apply(lambda x: 1 if x == "SUCCESS" else 0)

        def check_internal_ip(ip_str):
            if pd.isna(ip_str) or ip_str == "None": 
                return -1
            try:
                return 1 if ipaddress.ip_address(str(ip_str)).is_private else 0
            except ValueError:
                return -1
                
        df['is_internal_ip'] = df.get('source_ip', pd.Series([-1]*len(df))).apply(check_internal_ip)

        # ==========================================
        # PHASE 2: ADVANCED BEHAVIORAL ANALYTICS
        # ==========================================
        
        # Context & Geography
        df['geo_country'] = df.get('geo_country', 'Unknown').fillna('Unknown')
        
        # Historical User Baselines (Using duplicated to flag if this is new behavior)
        df['is_new_ip_for_user'] = (~df.duplicated(subset=['user_id', 'source_ip'])).astype(int)
        df['is_new_device_for_user'] = (~df.duplicated(subset=['user_id', 'browser_type', 'os_type'])).astype(int)

        # Factorize strings to numerical IDs for blazing fast rolling calculations
        df['ip_num'] = pd.factorize(df.get('source_ip', '0.0.0.0'))[0]
        df['resource_num'] = pd.factorize(df.get('resource', 'Unknown'))[0]

        # Velocity Metrics: Counts and Error Rates (Simplified for demo dataset)
        df['api_call_count_1m'] = df.groupby('user_id')['user_id'].transform("count").fillna(1)

        df['failed_actions_5m'] = df.groupby('user_id')['login_result_success'].transform(
            lambda x: (x == 0).cumsum()
        )
        df['total_actions_5m'] = df.groupby('user_id')['login_result_success'].transform("count")
        df['error_rate_5m'] = (df['failed_actions_5m'] / df['total_actions_5m']).fillna(0.0)

        # Read vs Write Ratio (Reconnaissance detection)
        df['is_read_action'] = df['action'].astype(str).str.contains('Get|List|Describe|Read', case=False).astype(int)
        df['is_write_action'] = (~df['is_read_action'].astype(bool)).astype(int)
        
        reads_1h = df.groupby('user_id')['is_read_action'].cumsum()
        writes_1h = df.groupby('user_id')['is_write_action'].cumsum()
        df['read_vs_write_ratio'] = (reads_1h / writes_1h.replace(0, 1)).fillna(0.0)

        # Scope & Escalation: Unique Resources and IPs accessed
        df["unique_resources_accessed"] = (
            df.groupby("user_id")["resource_num"]
            .transform("nunique")
            .fillna(1)
        )

        df["unique_ips_last_24h"] = (
            df.groupby("user_id")["ip_num"]
            .transform("nunique")
            .fillna(1)
        )
        
        # High-sensitivity actions in the last 24h
        df['is_privileged'] = df['action_sensitivity_score'].apply(lambda x: 1 if x >= 3 else 0)
        df["privileged_actions_last_24h"] = (
            df.groupby("user_id")["is_privileged"]
            .cumsum()
        )

        # Reset index back to normal integers
        df = df.reset_index(drop=True)

        # ==========================================
        # ENCODING & TARGET ISOLATION
        # ==========================================
        
        # Drop columns used for temporary math or raw strings too noisy for ML
        columns_to_drop = [
            'source_ip', 'destination_ip', 'resource', 'event_type', 'status', 'user_agent',
            'failed_actions_5m', 'total_actions_5m', 'ip_num', 'resource_num', 
            'is_read_action', 'is_write_action', 'is_privileged'
        ]
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

        # 1. Isolate Target Variables (y) to prevent Data Leakage
        y_cols = [col for col in self.target_columns if col in df.columns]
        y = df[y_cols].copy()
        
        # Safely map severity text if it exists
        if 'severity' in y.columns and y['severity'].dtype == 'object':
            y['severity'] = y['severity'].map({"INFO": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}).fillna(0)

        # 2. Isolate Feature Variables (X)
        X = df.drop(columns=y_cols)

        # 3. Label Encode ALL categorical strings in X
        categorical_cols = ['source_cloud', 'action', 'user_id', 'device_compliant_status', 'browser_type', 'os_type', 'geo_country']
        
        for col in categorical_cols:
            if col in X.columns:
                X[col] = X[col].fillna("UNKNOWN_VALUE").astype(str)
                if is_training:
                    X[col] = self.label_encoders[col].fit_transform(X[col])
                else:
                    X[col] = X[col].apply(lambda val: val if val in self.label_encoders[col].classes_ else "UNKNOWN_VALUE")
                    if "UNKNOWN_VALUE" not in self.label_encoders[col].classes_:
                        self.label_encoders[col].classes_ = np.append(self.label_encoders[col].classes_, "UNKNOWN_VALUE")
                    X[col] = self.label_encoders[col].transform(X[col])

        logger.info(f"Engineered {len(X.columns)} features for {len(X)} logs. Targets isolated: {list(y.columns)}")
        
        # ==========================================
        # OPTIONAL: EXPORT TO CSV
        # ==========================================
        if export_csv:
            X.to_csv(output_path_X, index=False)
            y.to_csv(output_path_y, index=False)
            logger.info(f"Successfully exported feature matrix to '{output_path_X}' and targets to '{output_path_y}'")

        # Returns standard Scikit-Learn format: X (Features), y (Targets)
        return X, y