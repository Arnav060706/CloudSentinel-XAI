"""
DEMO 2 — Identity-Stitching Graph Engine
Shows: three events that look like DIFFERENT principals, across THREE clouds,
get stitched into ONE actor whose lifetime cloud footprint = {AWS, AZURE, GCP}.
Also shows the ambiguity guard refusing to over-merge.
Run:  PYTHONPATH=. python demo/demo_2_graph.py
"""
from demo._util import banner, step, show, check, done
from app.services.graph_engine import MultiCloudGraphEngine

banner("DEMO 3 — IDENTITY STITCHING  (3 principals across 3 clouds -> 1 actor)")
g = MultiCloudGraphEngine()

def ev(principal, cloud, ts, ua="aws-cli", ver="2.13", ip="203.0.113.45",
       ptype="IAMUser", proxy=False, **extra):
    e = {"principal": principal, "source_cloud": cloud, "timestamp": ts,
         "ua_family": ua, "ua_version": ver, "source_ip": ip,
         "principal_type": ptype, "is_known_proxy_or_tor": proxy}
    e.update(extra); return e

step("Event 1 — alice logs into AWS")
eid1, win1, new1, method1, clouds1 = g.process_event(
    ev("alice", "AWS", "2026-06-10T08:00:00Z"))
show("entity_id", eid1); show("resolution_method", method1)
show("lifetime_clouds", clouds1)
check("first event creates a new entity", new1 is True)

step("Event 2 — alice CREATES a new IAM user 'svc-07' (provenance recorded)")
g.process_event(ev("alice", "AWS", "2026-06-10T08:01:00Z",
                   created_principal_name="svc-07"))

step("Event 3 — 'svc-07' acts in GCP (different principal, different cloud)")
eid3, win3, new3, method3, clouds3 = g.process_event(
    ev("svc-07", "GCP", "2026-06-10T08:02:00Z", ua="google-api"))
show("entity_id", eid3); show("resolution_method", method3)
check("svc-07 stitched to alice via CREATION PROVENANCE (not a new actor)",
      eid3 == eid1 and method3 == "provenance")

step("Event 4 — same actor appears in AZURE")
eid4, win4, new4, method4, clouds4 = g.process_event(
    ev("alice", "AZURE", "2026-06-10T08:03:00Z"))
show("resolution_method", method4)
show("LIFETIME cloud footprint for this actor", clouds4)
check("one actor now spans all 3 clouds", set(clouds4) == {"AWS", "AZURE", "GCP"})

step("Ambiguity / non-merge — a genuinely different user (different toolchain)")
eidX, *_ , methodX, cloudsX = g.process_event(
    ev("carol-unrelated", "AWS", "2026-06-10T08:04:00Z",
       ua="Mozilla-Firefox", ver="121.0", ip="198.51.100.9", ptype="AssumedRole"))
show("resolution_method", methodX)
check("a user with a distinct fingerprint is NOT merged into alice's actor",
      eidX != eid1)
print("   note: stitching is fingerprint-based — a user sharing alice's exact")
print("         toolchain (same UA family+version+type) can legitimately merge;")
print("         this is the tunable precision/recall tradeoff of Tier-2 fuzzy fusion.")

stats = g.get_entity_stats()
show("total known entities", stats.get("known_entities"))
done("DEMO 3: Identity-Stitching Graph Engine")