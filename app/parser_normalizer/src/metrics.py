from prometheus_client import Counter, Histogram

LOGS_RECEIVED = Counter('logs_received_total', 'Total number of logs received')
LOGS_NORMALIZED = Counter('logs_normalized_total', 'Total successfully normalized logs')
LOGS_FAILED = Counter('logs_failed_validation_total', 'Logs that failed validation')
PROCESS_TIME = Histogram('parser_processing_time_seconds', 'Time spent processing a log')