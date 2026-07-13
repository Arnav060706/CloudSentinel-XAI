"""
validate_realism.py — Statistical fidelity check: synthetic vs REAL company logs
Compares key distributions so you can show a reviewer the synthetic data is
calibrated to reality, not arbitrary. Prints a side-by-side table.
"""
import json, glob
from collections import Counter

def load(files):
    out=[]
    for f in files: out += json.load(open(f))
    return out

def aws_action(r): return r.get("eventName")
def az_action(r):  return r.get("operationName")
def gcp_action(r): return r.get("protoPayload",{}).get("methodName")

real_aws = json.load(open("dataset/real/aws_iam_logs.json"))
real_az  = json.load(open("dataset/real/azure_iam_logs.json"))
real_gcp = json.load(open("dataset/real/gcp_iam_logs.json"))
syn_aws = json.load(open("dataset/merged/aws_logs.json"))
syn_az  = json.load(open("dataset/merged/azure_logs.json"))
syn_gcp = json.load(open("dataset/merged/gcp_logs.json"))

def pct_anom(logs): 
    a=sum(1 for r in logs if r.get("ml_labels",{}).get("anomaly_flag")); 
    return 100*a/len(logs) if logs else 0
def uniq_ips(logs, key):
    return len(set(key(r) for r in logs))
def field_overlap(real, syn, key):
    rs=set(filter(None,(key(r) for r in real))); ss=set(filter(None,(key(r) for r in syn)))
    return len(rs & ss), len(rs)

print(f"{'Metric':<42}{'REAL':>12}{'SYNTHETIC':>12}")
print("-"*66)
print(f"{'AWS records':<42}{len(real_aws):>12}{len(syn_aws):>12}")
print(f"{'Azure records':<42}{len(real_az):>12}{len(syn_az):>12}")
print(f"{'GCP records':<42}{len(real_gcp):>12}{len(syn_gcp):>12}")
print(f"{'% anomalous (AWS)':<42}{pct_anom(real_aws):>11.1f}%{pct_anom(syn_aws):>11.1f}%")
print(f"{'% anomalous (Azure)':<42}{pct_anom(real_az):>11.1f}%{pct_anom(syn_az):>11.1f}%")
print(f"{'% anomalous (GCP)':<42}{pct_anom(real_gcp):>11.1f}%{pct_anom(syn_gcp):>11.1f}%")

# vocabulary overlap: are synthetic actions/IPs drawn from the real universe?
ov, tot = field_overlap(real_aws, syn_aws, lambda r:r.get("sourceIPAddress"))
print(f"{'AWS source IPs from real pool':<42}{tot:>12}{ov:>9}/{tot}")
ov, tot = field_overlap(real_aws, syn_aws, aws_action)
print(f"{'AWS event types seen in real logs':<42}{tot:>12}{ov:>9}/{tot}")
ov, tot = field_overlap(real_gcp, syn_gcp, gcp_action)
print(f"{'GCP methods seen in real logs':<42}{tot:>12}{ov:>9}/{tot}")

# threat category coverage
real_cats = set(r.get("ml_labels",{}).get("threat_category") for r in real_aws+real_az+real_gcp)
syn_cats  = set(r.get("ml_labels",{}).get("threat_category") for r in syn_aws+syn_az+syn_gcp)
print("-"*66)
print(f"Real threat categories : {sorted(c for c in real_cats if c)}")
print(f"Synthetic categories   : {sorted(c for c in syn_cats if c)}")
print(f"Synthetic ADDS (beyond real sample): {sorted(syn_cats - real_cats - {None})}")