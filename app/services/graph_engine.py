# app/services/graph_engine.py
"""
MultiCloudGraphEngine — Sliding-Window Stateful Identity Stitching Engine
=========================================================================

ARCHITECTURAL SEPARATION INTRODUCED IN THIS VERSION — READ THIS FIRST:

  Earlier versions conflated two genuinely different concepts under one
  data structure: "is this event recent enough to matter for live risk
  scoring" (a 60-second concern) and "do we still know who this actor is"
  (a campaign-lifetime concern, which should NOT reset just because
  nothing happened for a minute). Conflating them created two real bugs,
  both surfaced by asking "what if an APT paces its steps 2 minutes
  apart instead of within 60 seconds":

    BUG A (identity resurrection): when an entity's deque fully drained
    (all its events aged out of the 60s window with no new activity),
    the old code deleted the entity's identity mapping and its anchor
    entirely. A RETURNING session for the exact same actor, arriving any
    time after that drain, would resolve to a BRAND NEW entity_id instead
    of the original one — silently breaking stitching for any campaign
    paced slower than the window, which is a very plausible evasion
    strategy for an attacker who knows a system like this exists.

    BUG B (cross-cloud amplification never fires for paced campaigns):
    the Hawkes engine's diversity multiplier was being computed from
    ONLY the clouds present in the current 60-second window. Two steps
    of the same campaign 2 minutes apart, in different clouds, are never
    simultaneously "in window" together — so the cross-cloud multiplier,
    the system's headline detection mechanism, silently never triggers
    for anything paced slower than the window.

  THE FIX: identity (entity_anchors, identity_map, and a NEW lifetime
  cloud-footprint set) is now permanent for the life of the process
  (subject only to an EXPLICIT, disclosed retention/purge operation you
  call yourself — see purge_stale_entities). The sliding deque
  (_graph_registry) remains genuinely transient and window-scoped, and
  CAN legitimately be empty for a known, still-tracked entity — an empty
  window means "nothing from this actor in the last 60 seconds," not
  "we no longer know who this is."

  This also simplifies the eviction path: because identity is no longer
  destroyed when a window drains, there is no need for the reference-
  counting bookkeeping that a prior version used to decide when to clean
  up identity_map entries. Eviction is now a plain popleft with no
  per-key accounting at all — still O(1) per evicted event, now with
  less code and one less category of bug to introduce by accident.

Implements the tiered identity correlation model for CloudSentinel-XAI:

  TIER 1 — Federation Join (fast path, O(1)):
  If the normalized event carries a federation lineage field
  (sourceIdentity / SAML assertion / OIDC subject), that field is used as
  a direct join key. Covers the phase of a kill chain where the attacker
  is still operating through a legitimate federated session.
  CAVEAT (unchanged from before): this tier's coverage is conditional on
  the target org having configured sts:SourceIdentity enforcement (AWS)
  or an equivalent federation-passthrough setting. Frequently absent even
  for a legitimate user's first login. When absent, falls through to
  Tier 1.5 / Tier 2.

  TIER 1.5 — Creation Provenance (deterministic, O(1)):
  Cloud audit logs record, as a plain stated fact, who performed a
  CreateUser / CreateServiceAccount / CreateRole action. If the event
  indicates it just created a new principal ("created_principal_name"),
  and the creator is already resolved to an entity, the new principal is
  linked to that SAME entity directly — no similarity scoring, no
  ambiguity. This is the correct mechanism for "Alice, Bob, and Carol are
  all active — which of them does this new service account belong to?"
  The log states the answer; fuzzy inference must never override a fact
  that's already in the data.

  TIER 2 — Fuzzy Fusion (fallback: orphaned principal, no federation
  lineage, no creation-provenance record). Four signals fused:
    - s_ua    : UA-family + version similarity           (weight 0.45)
    - s_proxy : shared proxy/Tor infrastructure alignment (weight 0.25)
    - s_type  : principal-type consistency                (weight 0.20)
    - s_ip    : exact source-IP match, CLEAN TRAFFIC ONLY  (weight 0.10)
  An ambiguity-margin check refuses to merge when the top two candidates
  are too close to call (e.g. several simultaneously active legitimate
  users with similar tooling).

  ON s_ip: deliberately small and capped. It exists because an
  unsophisticated attacker who never rotates IP still deserves to be
  caught on that basis — but 0.10 (or 0.10 + the 0.20 type match = 0.30)
  can never alone reach the default tau of 0.65, so it can only
  corroborate an already-plausible match, never manufacture one alone.
  Only applies when BOTH sides are non-proxy traffic — matching two
  Tor/VPN exit IPs is coincidence (shared exit nodes across unrelated
  sessions), not correlation.

Complexity guarantees
---------------------
  Insertion (per event)     : O(1) amortized
  Window eviction            : O(1) amortized per evicted event — plain
    deque popleft, no per-key bookkeeping (identity is no longer torn
    down on window drain, so there is nothing to account for here).
  Tier 1 / Tier 1.5 / fast-path resolution : O(1) each (hashmap lookups)
  Tier 2 slow-path            : O(E) where E = number of KNOWN entities
    (not just currently-active ones, since identity now persists across
    idle periods) — disclosed, not claimed as O(1).
  purge_stale_entities()      : O(V) where V = total tracked identity
    keys. This is an EXPLICIT, infrequent, caller-invoked maintenance
    operation, not part of the per-event hot path — an O(V) scan here is
    fine and is clearly scoped as a maintenance operation, unlike the
    earlier bug where an O(V)/O(n) scan was hidden inside the hot
    per-event eviction path and silently broke the O(1) claim.

Thread safety
-------------
  A reentrant lock (RLock) guards all mutations.
"""

import time
import threading
import logging
import uuid
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class MultiCloudGraphEngine:
    """
    Sliding-window directed graph engine for cross-cloud identity stitching,
    with campaign-lifetime (not just window-scoped) cross-cloud tracking.

    Parameters
    ----------
    window_horizon_seconds : int
        Length of the sliding time window used for RISK SCORING recency —
        i.e. what risk_engine.py's Hawkes intensity sees as "currently
        active" for its temporal-decay sum. Default: 60 seconds.
        IMPORTANT: this window no longer governs identity persistence or
        cross-cloud tracking at all (see module docstring) — an entity is
        NOT forgotten and its cross-cloud footprint is NOT reset just
        because its window emptied.
    similarity_threshold : float
        Minimum fuzzy-fusion score (tau) for Tier 2 merges. Default: 0.65.
    ambiguity_margin : float
        Minimum score gap between the best and second-best Tier 2
        candidates before a merge is accepted. Default: 0.05.
    """

    _UA_WEIGHT    = 0.45
    _PROXY_WEIGHT = 0.25
    _TYPE_WEIGHT  = 0.20
    _IP_WEIGHT    = 0.10

    _CREATION_ACTIONS: Set[str] = {
        "CreateUser", "CreateRole", "CreateServiceAccount",
        "Add application", "Add service principal", "Add user",
        "google.iam.admin.v1.CreateServiceAccount",
    }

    # Clouds that should be treated as the same provider for cross-cloud
    # counting purposes (Entra ID is Azure's identity plane, not a 4th
    # cloud) — mirrors the normalization risk_engine.py already applies.
    _CLOUD_ALIASES = {"ENTRA-ID": "AZURE"}

    # principal_type is cloud-native vocabulary, not a normalized concept:
    # AWS says "IAMUser"/"AssumedRole", Azure says "User"/"ServicePrincipal",
    # GCP says "User"/"ServiceAccount". Comparing these strings directly (as
    # the previous version did) means a genuine human attacker's own AWS leg
    # NEVER earns type-match credit against their own Azure/GCP legs of the
    # SAME campaign, purely due to vocabulary mismatch -- actively hurting
    # the exact cross-cloud stitching this engine exists for. Normalize into
    # the same human/automation concept risk_engine.py's _classify_principal
    # already uses, so "IAMUser" and "User" (both human) correctly match.
    _AUTOMATION_PRINCIPAL_TYPES = {
        "ASSUMEDROLE", "SERVICEACCOUNT", "SERVICEPRINCIPAL", "ROLE",
        "AWSSERVICE", "AWSACCOUNT",
    }
    _HUMAN_PRINCIPAL_TYPES = {"IAMUSER", "USER", "ROOT", "FEDERATEDUSER"}

    # Maps the internal resolution-method string to the coarse tier label used
    # in the merge audit log (Phase 1a).
    _AUDIT_TIER = {
        "federation": "1-federation",
        "provenance": "1.5-provenance",
        "fast_path": "fast_path",
        "fuzzy_merge": "2-fuzzy",
        "new_entity": "new",
        "new_entity_ambiguous": "new",
    }

    def __init__(
        self,
        window_horizon_seconds: int = 60,
        similarity_threshold: float = 0.65,
        ambiguity_margin: float = 0.05,
        tier2_lookback_seconds: float = 10800,
        merge_audit_log: Optional[list] = None,
    ):
        # Phase 1a: optional diagnostic. When a caller passes a list here, the
        # engine appends one record per entity assignment (see _audit_assignment)
        # so a downstream script can score merge correctness against ground
        # truth. Purely additive — when None (the default) there is ZERO behavior
        # change and no per-event overhead beyond a single `is None` check.
        self.merge_audit_log = merge_audit_log
        self.window_horizon = window_horizon_seconds
        self.tau = similarity_threshold
        self.ambiguity_margin = ambiguity_margin
        # Tier 2 candidate scan is bounded to entities last active within
        # this many seconds. Without this, campaigns days/weeks apart could
        # still merge purely because they share generic traits (UA family,
        # proxy flag, principal type) with no time relationship at all --
        # confirmed empirically as part of the over-merging bug (148 real
        # identities collapsed into 3 entities). Default 10800s (3h) is
        # derived from generate_attacks.py's PACE table: "slow" pacing gaps
        # up to 5400s (1.5h) between consecutive steps of the SAME
        # campaign, so 3h gives ~2x safety margin without being so long it
        # re-admits the unrelated-campaigns-weeks-apart failure mode. This
        # bounds Tier-2 MATCHING eligibility only -- it does not affect
        # identity permanence (entity_anchors, lifetime clouds, and
        # identity_map entries are still never deleted except via the
        # explicit purge_stale_entities() call).
        self.tier2_lookback = tier2_lookback_seconds

        # entity_id -> deque of {"arrival_time": float, "data": dict}
        # Transient, window-scoped. CAN legitimately be empty for a
        # perfectly valid, still-tracked entity — see module docstring.
        self._graph_registry: Dict[str, deque] = {}

        # entity_id -> founding event. Permanent for the life of the
        # process (or until purge_stale_entities removes it explicitly).
        # Stable Tier 2 comparison target, independent of window state.
        self._entity_anchors: Dict[str, dict] = {}

        # entity_id -> set of normalized cloud names EVER seen for this
        # entity, across its ENTIRE lifetime — NOT scoped to the 60s
        # window. This is what fixes the "2-minute-paced APT" problem:
        # the cross-cloud diversity multiplier in risk_engine.py should
        # be driven by this set, not by whichever clouds happen to be
        # inside the live risk window at any one instant.
        self._entity_lifetime_clouds: Dict[str, Set[str]] = {}

        # entity_id -> unix timestamp of the most recent event seen. Used by
        # purge_stale_entities() AND (as of the tier2_lookback fix above) by
        # _resolve_entity_id()'s Tier-2 scan, to bound which entities are
        # even eligible for a fuzzy-fusion match.
        self._entity_last_seen: Dict[str, float] = {}

        # any identity key (principal string OR federation key) -> entity_id.
        # Permanent, same lifetime as entity_anchors.
        self._identity_map: Dict[str, str] = {}

        # created_principal_name -> creator's entity_id. Permanent
        # (Tier 1.5 provenance table) — same reasoning as before.
        self._provenance_map: Dict[str, str] = {}

        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def process_event(
        self, normalized_event: dict
    ) -> Tuple[str, List[dict], bool, str, List[str]]:
        """
        Main entry point. Resolves the entity cluster for an incoming
        normalized event, appends it to the sliding window, evicts stale
        window entries, updates the entity's lifetime cloud footprint, and
        returns everything downstream risk scoring needs.

        Parameters
        ----------
        normalized_event : dict
            Must contain at minimum: "principal", "ua_family", "ua_version",
            "is_known_proxy_or_tor", "principal_type", "source_ip",
            "source_cloud", "timestamp".
            Optional: "federation_id", "created_principal_name".

        Returns
        -------
        entity_id : str
        active_events : List[dict]
            Events currently inside the 60-second RISK window for this
            entity (for feeding the Hawkes temporal-decay sum).
        is_new_entity : bool
        resolution_method : str
            "federation" | "provenance" | "fast_path" | "fuzzy_merge" |
            "new_entity_ambiguous" | "new_entity"
        lifetime_clouds : List[str]
            Every distinct normalized cloud provider this entity has EVER
            touched, across its whole tracked lifetime — NOT limited to
            the current 60-second window. Pass this (not a value derived
            from active_events) into HawkesRiskEngine.calculate_intensity's
            lifetime_clouds parameter so the cross-cloud multiplier fires
            correctly regardless of how slowly an attacker paces their
            campaign across clouds.
        """
        with self._lock:
            now = time.time()
            entity_id, is_new, method, similarity = self._resolve_entity_id(normalized_event, now)
            if self.merge_audit_log is not None:
                self._audit_assignment(normalized_event, entity_id, method, similarity)
            self._register_provenance_if_applicable(normalized_event, entity_id)
            active_events = self._update_window(entity_id, normalized_event, now)
            lifetime_clouds = sorted(self._entity_lifetime_clouds.get(entity_id, set()))
            return entity_id, active_events, is_new, method, lifetime_clouds

    def _audit_assignment(self, event: dict, entity_id: str, method: str,
                          similarity: Optional[float]) -> None:
        """Phase 1a: record one merge/assignment decision. Captures the entity's
        OTHER principals as they stood BEFORE this event, so a merge that pulls a
        second real identity into an existing entity is visible in the log. Only
        called when merge_audit_log is enabled."""
        principal = event.get("principal", "")
        principals_so_far = sorted(
            p for p, eid in self._identity_map.items()
            if eid == entity_id and p != principal
        )
        self.merge_audit_log.append({
            "event_id": event.get("event_id"),
            "tier": self._AUDIT_TIER.get(method, method),
            "method": method,
            "similarity_score": similarity,
            "entity_id": entity_id,
            "event_principal": principal,
            "entity_principals_so_far": principals_so_far,
        })

    def get_entity_stats(self) -> dict:
        """Snapshot of current graph state, for dashboards/debugging."""
        with self._lock:
            return {
                "known_entities": len(self._entity_anchors),
                "entities_with_active_window": sum(
                    1 for q in self._graph_registry.values() if q
                ),
                "tracked_identity_keys": len(self._identity_map),
                "tracked_provenance_records": len(self._provenance_map),
                "window_horizon_seconds": self.window_horizon,
                "similarity_threshold": self.tau,
                "ambiguity_margin": self.ambiguity_margin,
                "entity_window_sizes": {
                    eid: len(q) for eid, q in self._graph_registry.items()
                },
                "entity_lifetime_cloud_counts": {
                    eid: len(clouds)
                    for eid, clouds in self._entity_lifetime_clouds.items()
                },
            }

    def entity_lifetime_footprints(self) -> Dict[str, list]:
        """Snapshot of each entity's accumulated lifetime cloud set, as sorted
        lists. Used to seed RiskEngine per-identity baselines from a benign
        warmup pass (Phase 1 fix) without reaching into private state."""
        with self._lock:
            return {eid: sorted(clouds)
                    for eid, clouds in self._entity_lifetime_clouds.items()}

    def reset_windows(self) -> None:
        """Clear the transient 60s sliding windows while KEEPING identity,
        lifetime-cloud, and provenance state. Needed when a benign warmup pass
        (to learn baselines) is followed by a scoring pass whose event times
        precede the warmup's last event: without this, stale warmup events left
        in a window would be counted (at negative age -> full weight) against the
        scored events. Identity persistence is intentionally untouched."""
        with self._lock:
            for q in self._graph_registry.values():
                q.clear()

    def purge_stale_entities(self, max_idle_seconds: float) -> int:
        """
        EXPLICIT, caller-invoked maintenance operation. Removes entities
        that have had no activity for longer than max_idle_seconds,
        cleaning up their anchor, lifetime cloud set, last-seen record,
        and any identity_map / provenance_map entries pointing to them.

        This is the disclosed answer to "identity and lifetime cloud
        tracking now grow unboundedly for the life of the process" — call
        this periodically (e.g. hourly, from a background scheduler) with
        a generous max_idle_seconds (e.g. 86400 for a day) to bound memory
        growth without breaking pace-independent cross-cloud detection for
        any campaign shorter than max_idle_seconds.

        Complexity: O(V) where V = total tracked identity keys. This is
        fine here because it is an infrequent, explicit, disclosed
        operation — NOT hidden inside the per-event hot path the way an
        earlier bug in this file hid an O(V) scan inside eviction.

        Returns the number of entities purged.
        """
        with self._lock:
            now = time.time()
            stale_entity_ids = [
                eid for eid, last_seen in self._entity_last_seen.items()
                if (now - last_seen) > max_idle_seconds
            ]

            for eid in stale_entity_ids:
                self._entity_anchors.pop(eid, None)
                self._entity_lifetime_clouds.pop(eid, None)
                self._entity_last_seen.pop(eid, None)
                self._graph_registry.pop(eid, None)

                dangling_identity_keys = [
                    k for k, v in self._identity_map.items() if v == eid
                ]
                for k in dangling_identity_keys:
                    self._identity_map.pop(k, None)

                dangling_provenance_keys = [
                    k for k, v in self._provenance_map.items() if v == eid
                ]
                for k in dangling_provenance_keys:
                    self._provenance_map.pop(k, None)

            if stale_entity_ids:
                logger.info(
                    "purge_stale_entities: removed %d entities idle > %.0fs",
                    len(stale_entity_ids), max_idle_seconds,
                )
            return len(stale_entity_ids)

    # ------------------------------------------------------------------ #
    # Tier 1 — Federation join                                             #
    # ------------------------------------------------------------------ #

    def _tier1_federation_key(self, event: dict) -> Optional[str]:
        """
        Extracts the upstream IdP linkage key if present.
        (See prior version's docstring for full field-mapping detail —
        unchanged: AWS sourceIdentity / IAM Identity Center session-name
        convention, Azure claims.sub/oid, GCP principalSubject. Frequently
        absent — see caveats in module docstring.)
        """
        fed_id = event.get("federation_id")
        if fed_id and str(fed_id).strip() not in ("", "None", "null"):
            return str(fed_id).strip()
        return None

    # ------------------------------------------------------------------ #
    # Tier 1.5 — Creation provenance                                       #
    # ------------------------------------------------------------------ #

    def _register_provenance_if_applicable(self, event: dict, acting_entity_id: str) -> None:
        """
        Records a deterministic link from a newly created principal's name
        to the entity that created it — see module docstring for why this
        must take priority over fuzzy inference. Permanent; not tied to
        window state (unchanged from prior version).
        """
        action = event.get("action")
        created_name = event.get("created_principal_name")

        if created_name and (action in self._CREATION_ACTIONS or created_name):
            self._provenance_map[created_name] = acting_entity_id
            logger.debug(
                "Provenance recorded: '%s' created by entity %s (action=%s)",
                created_name, acting_entity_id, action,
            )

    # ------------------------------------------------------------------ #
    # Tier 2 — Fuzzy fusion similarity                                     #
    # ------------------------------------------------------------------ #

    # Sentinel values meaning "we don't actually know this" -- matching on
    # one of these must NEVER earn similarity credit. A signal is only
    # evidence of shared identity if it's informative; two unrelated people
    # who both used an unrecognized tool are not thereby similar to each
    # other, they're both just unidentified. Bug fixed here: the previous
    # code only excluded the literal string "Unknown", but
    # enrichment.py's parse_user_agent() fallback returns ("Other",
    # "Unknown") for anything that doesn't match a known UA pattern (curl,
    # python-requests, Go-http-client, etc. -- exactly the tooling most
    # attack traffic in this dataset uses) -- so "Other" == "Other" was
    # silently passing the guard and awarding the full UA weight for a
    # match on pure ignorance. Confirmed empirically: this collapsed 148
    # distinct real identities into 3 entities on attacks_fast.
    _UNINFORMATIVE_UA_FAMILY = {"Unknown", "Other"}
    _UNINFORMATIVE_VALUES = {"Unknown", "Other", None, ""}

    def _calculate_fuzzy_similarity(self, event_a: dict, event_b: dict) -> float:
        """
        Four-signal fuzzy similarity, weights summing to 1.0.

        CHANGED from the previous version: three places were awarding
        similarity credit for two sides both lacking information, rather
        than for two sides genuinely matching on a specific, identifying
        value. Fixed all three (see _UNINFORMATIVE_UA_FAMILY /
        _UNINFORMATIVE_VALUES above and inline comments below) -- this was
        the root cause of the over-merging bug, not a threshold/weight
        tuning issue.
        """
        score = 0.0

        KNOWN_AUTOMATION_FAMILIES = {
            "Boto3", "aws-cli", "aws-sdk-go", "Terraform",
            "azure-cli", "google-cloud-sdk", "pulumi",
        }

        ua_a = event_a.get("ua_family", "Unknown")
        ua_b = event_b.get("ua_family", "Unknown")

        if ua_a == ua_b and ua_a not in self._UNINFORMATIVE_UA_FAMILY:
            score += self._UA_WEIGHT * 0.70
            ver_a = event_a.get("ua_version")
            ver_b = event_b.get("ua_version")
            # Version sub-bonus also needs its own guard -- matching on two
            # "Unknown" versions is exactly the same uninformative-match bug.
            if ver_a == ver_b and ver_a not in self._UNINFORMATIVE_VALUES:
                score += self._UA_WEIGHT * 0.30
        elif ua_a in KNOWN_AUTOMATION_FAMILIES and ua_b in KNOWN_AUTOMATION_FAMILIES:
            score += self._UA_WEIGHT * 0.20

        proxy_a = bool(event_a.get("is_known_proxy_or_tor", False))
        proxy_b = bool(event_b.get("is_known_proxy_or_tor", False))

        # Only a confirmed True/True match is informative (both are known
        # Tor/hosting traffic -- genuinely rare). Removed the previous
        # "proxy_a == proxy_b" branch, which also fired for False/False --
        # since most traffic (benign AND most attack traffic here, given
        # the hosting/foreign infra ASN-keyword gap) is proxy=False, that
        # branch was awarding corroborating credit to nearly every pair.
        if proxy_a and proxy_b:
            score += self._PROXY_WEIGHT * 1.0

        if not proxy_a and not proxy_b:
            ip_a = event_a.get("source_ip")
            ip_b = event_b.get("source_ip")
            if ip_a and ip_b and ip_a == ip_b:
                score += self._IP_WEIGHT

        type_a = self._normalize_principal_type(event_a.get("principal_type"))
        type_b = self._normalize_principal_type(event_b.get("principal_type"))
        if type_a == type_b and type_a is not None:
            score += self._TYPE_WEIGHT

        return round(score, 4)

    def explain_fuzzy_similarity(self, event_a: dict, event_b: dict) -> dict:
        """Diagnostic sibling of _calculate_fuzzy_similarity: returns the SAME
        total plus a per-component breakdown (which signal contributed how much),
        so an audit can say exactly which signal over-credited a wrong merge
        (Phase 1b). Read-only; mirrors the scoring logic above component-for-
        component so the two cannot silently diverge in weighting."""
        comp = {"ua": 0.0, "proxy": 0.0, "ip": 0.0, "type": 0.0}

        KNOWN_AUTOMATION_FAMILIES = {
            "Boto3", "aws-cli", "aws-sdk-go", "Terraform",
            "azure-cli", "google-cloud-sdk", "pulumi",
        }
        ua_a = event_a.get("ua_family", "Unknown")
        ua_b = event_b.get("ua_family", "Unknown")
        if ua_a == ua_b and ua_a not in self._UNINFORMATIVE_UA_FAMILY:
            comp["ua"] += self._UA_WEIGHT * 0.70
            ver_a, ver_b = event_a.get("ua_version"), event_b.get("ua_version")
            if ver_a == ver_b and ver_a not in self._UNINFORMATIVE_VALUES:
                comp["ua"] += self._UA_WEIGHT * 0.30
        elif ua_a in KNOWN_AUTOMATION_FAMILIES and ua_b in KNOWN_AUTOMATION_FAMILIES:
            comp["ua"] += self._UA_WEIGHT * 0.20

        proxy_a = bool(event_a.get("is_known_proxy_or_tor", False))
        proxy_b = bool(event_b.get("is_known_proxy_or_tor", False))
        if proxy_a and proxy_b:
            comp["proxy"] += self._PROXY_WEIGHT * 1.0
        if not proxy_a and not proxy_b:
            ip_a, ip_b = event_a.get("source_ip"), event_b.get("source_ip")
            if ip_a and ip_b and ip_a == ip_b:
                comp["ip"] += self._IP_WEIGHT

        type_a = self._normalize_principal_type(event_a.get("principal_type"))
        type_b = self._normalize_principal_type(event_b.get("principal_type"))
        if type_a == type_b and type_a is not None:
            comp["type"] += self._TYPE_WEIGHT

        comp = {k: round(v, 4) for k, v in comp.items()}
        comp["total"] = round(sum(comp.values()), 4)
        return comp

    def _normalize_principal_type(self, raw_type: Optional[str]) -> Optional[str]:
        """Cloud-agnostic human/automation bucket -- see class docstring
        comment above _AUTOMATION_PRINCIPAL_TYPES for why this exists.
        Returns None for anything unrecognized, so (per the uninformative-
        match fix above) two unrecognized types never earn credit for
        "matching" on not being classifiable."""
        t = str(raw_type or "").upper()
        if t in self._AUTOMATION_PRINCIPAL_TYPES:
            return "automation"
        if t in self._HUMAN_PRINCIPAL_TYPES:
            return "human"
        return None

    # ------------------------------------------------------------------ #
    # Entity resolution                                                    #
    # ------------------------------------------------------------------ #

    def _resolve_entity_id(self, incoming_event: dict, now: float) -> Tuple[str, bool, str, Optional[float]]:
        """
        Resolution order: Tier 1 -> fast path -> Tier 1.5 -> Tier 2 (with
        ambiguity margin) -> new entity.

        Tier 2's scan considers every entity last active within
        self.tier2_lookback seconds of `now` (every key in _entity_anchors
        whose _entity_last_seen passes that bound), not just ones with a
        currently non-empty 60s window -- an entity whose window has
        drained is still a perfectly valid, resolvable identity, since its
        anchor is a fixed point in time. This is what makes a paced
        attacker whose window is momentarily empty between steps still
        resolvable. The lookback bound (separate from the window) is what
        stops that same reasoning from over-reaching into matching entities
        that are ACTUALLY unrelated, just because they share generic
        traits with no time relationship at all -- see tier2_lookback's
        docstring in __init__ for why this bound exists.
        """
        principal = incoming_event.get("principal", "")

        fed_key = self._tier1_federation_key(incoming_event)
        if fed_key:
            if fed_key not in self._identity_map:
                new_id = self._create_new_entity(fed_key, incoming_event)
                self._identity_map[principal] = new_id
                logger.debug("Tier1 new entity %s for federation key %s", new_id, fed_key)
                return new_id, True, "federation", None
            entity_id = self._identity_map[fed_key]
            self._identity_map[principal] = entity_id
            return entity_id, False, "federation", None

        if principal in self._identity_map:
            return self._identity_map[principal], False, "fast_path", None

        if principal in self._provenance_map:
            creator_entity_id = self._provenance_map[principal]
            if creator_entity_id in self._entity_anchors:
                self._identity_map[principal] = creator_entity_id
                logger.debug(
                    "Tier1.5 provenance match: '%s' linked to creator entity %s",
                    principal, creator_entity_id,
                )
                return creator_entity_id, False, "provenance", None

        scored_candidates: List[Tuple[float, str]] = []
        for entity_id, anchor in self._entity_anchors.items():
            # NOTE: no longer filtered by "does this entity have a
            # non-empty window right now" — see method docstring above.
            # NEW: bounded by tier2_lookback -- an entity idle longer than
            # this is not eligible for a Tier-2 match, regardless of how
            # similar it looks (see __init__ / method docstring for why).
            last_seen = self._entity_last_seen.get(entity_id, 0.0)
            if (now - last_seen) > self.tier2_lookback:
                continue
            sim = self._calculate_fuzzy_similarity(incoming_event, anchor)
            if sim >= self.tau:
                scored_candidates.append((sim, entity_id))

        if scored_candidates:
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            best_score, best_entity_id = scored_candidates[0]

            if len(scored_candidates) > 1:
                second_score, second_entity_id = scored_candidates[1]
                if (best_score - second_score) < self.ambiguity_margin:
                    new_id = self._create_new_entity(principal, incoming_event)
                    logger.warning(
                        "Ambiguous Tier2 match for '%s': top candidates %s "
                        "(%.3f) and %s (%.3f) within margin %.3f — "
                        "creating new entity instead of merging.",
                        principal, best_entity_id, best_score,
                        second_entity_id, second_score, self.ambiguity_margin,
                    )
                    return new_id, True, "new_entity_ambiguous", best_score

            self._identity_map[principal] = best_entity_id
            logger.debug(
                "Tier2 merged principal '%s' into entity %s (score=%.3f)",
                principal, best_entity_id, best_score,
            )
            return best_entity_id, False, "fuzzy_merge", best_score

        new_id = self._create_new_entity(principal, incoming_event)
        logger.debug(
            "New entity %s for orphaned principal '%s' (no candidates cleared tau=%.2f)",
            new_id, principal, self.tau,
        )
        return new_id, True, "new_entity", None

    def _create_new_entity(self, key: str, founding_event: dict) -> str:
        """
        Instantiates a new entity cluster. Anchor, lifetime cloud set, and
        last-seen record are all created here as PERMANENT records (only
        removed via explicit purge_stale_entities, never by window drain).
        """
        # NOT time.time_ns() -- confirmed empirically that this system's
        # clock resolution is far coarser than nanoseconds (1000 rapid
        # time.time_ns() calls returned a single identical value). Under
        # any reasonably fast event rate, that collapsed distinct new
        # entities created in quick succession onto the SAME dict key,
        # silently overwriting each other in _entity_anchors /
        # _graph_registry / _identity_map. uuid4 doesn't depend on clock
        # resolution or call rate at all.
        entity_id = f"entity_{uuid.uuid4().hex}"
        self._identity_map[key] = entity_id
        self._entity_anchors[entity_id] = founding_event
        self._graph_registry[entity_id] = deque()
        self._entity_lifetime_clouds[entity_id] = set()
        self._entity_last_seen[entity_id] = time.time()
        return entity_id

    # ------------------------------------------------------------------ #
    # Sliding window management                                            #
    # ------------------------------------------------------------------ #

    def _normalize_cloud(self, raw_cloud: Optional[str]) -> str:
        """Applies the same Entra-ID-as-Azure aliasing risk_engine.py uses."""
        c = str(raw_cloud or "UNKNOWN").upper()
        return self._CLOUD_ALIASES.get(c, c)

    def _update_window(self, entity_id: str, incoming_event: dict, now: float) -> List[dict]:
        """
        Appends the incoming event to the entity's RISK window and evicts
        window entries older than window_horizon_seconds. Also updates the
        entity's PERMANENT lifetime cloud footprint and last-seen timestamp
        — neither of which is affected by window eviction.

        Complexity
        ----------
        Append   : O(1)
        Eviction : O(1) amortized per evicted event — a plain popleft with
          NO per-key bookkeeping. This is simpler and more clearly correct
          than the previous ref-counting scheme, because identity is no
          longer torn down when the window drains, so there is nothing to
          account for at eviction time beyond removing the stale event
          itself.

        IMPORTANT: an empty deque after eviction is now a perfectly normal,
        valid state — it means "no window-relevant activity in the last
        window_horizon_seconds," not "this entity no longer exists." The
        entity's anchor, lifetime clouds, and identity mappings all remain
        intact regardless.
        """
        current_time = now
        queue = self._graph_registry[entity_id]

        queue.append({"arrival_time": current_time, "data": incoming_event})

        # Update PERMANENT, lifetime (not window-scoped) cross-cloud
        # footprint. This is the actual fix for pace-independent detection:
        # an entity that touched AWS at t=0 and GCP at t=120 (2 minutes
        # apart — outside any 60s window together) still ends up with
        # lifetime_clouds = {"AWS", "GCP"}, so risk_engine.py's diversity
        # multiplier will correctly see |C_active|=2 for it, regardless of
        # how the two events relate to the live risk window.
        cloud = self._normalize_cloud(incoming_event.get("source_cloud"))
        self._entity_lifetime_clouds.setdefault(entity_id, set()).add(cloud)
        self._entity_last_seen[entity_id] = current_time

        # Evict stale window entries — simple, no bookkeeping needed.
        while queue and (current_time - queue[0]["arrival_time"]) > self.window_horizon:
            queue.popleft()

        # NOTE: deliberately NOT deleting graph_registry[entity_id],
        # entity_anchors[entity_id], or identity_map entries when the
        # queue empties here. That was the root cause of the identity-
        # resurrection bug (see module docstring) — an empty window must
        # not mean "forget this entity." Cleanup of genuinely stale
        # entities is an explicit, disclosed, caller-invoked operation:
        # see purge_stale_entities().

        return [node["data"] for node in queue]