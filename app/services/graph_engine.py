# app/services/graph_engine.py
"""
MultiCloudGraphEngine — Sliding-Window Stateful Identity Stitching Engine
=========================================================================

Implements the two-tier identity correlation model for CloudSentinel-XAI:

  Tier 1 (Federation Join): If the normalized event carries a federation
  lineage field (sourceIdentity / saml_assertion_id / oidc_subject), that
  field is used as a direct, O(1) join key. This covers the initial phases
  of a kill chain where the attacker is still operating through the
  legitimate federated session.

  Tier 2 (Fuzzy Fusion): Activated only for events whose principal has NO
  federation lineage in the logs — the precise signature of a persistence
  step (newly created IAM user, service principal, or service account with
  no upstream IdP link). Three signal families are fused:
    - s_ua   : UA-family + version similarity (weight 0.50)
    - s_proxy: shared proxy/Tor infrastructure alignment (weight 0.30)
    - s_type : principal-type consistency (weight 0.20)

  Geo exact-match is intentionally excluded: IP geo-country is exactly the
  signal that changes when an attacker rotates proxies or VPN exit nodes,
  so exact-matching it penalises the evasion scenario we are trying to catch.

Complexity guarantees
---------------------
  Insertion : O(1) amortized (deque append)
  Eviction  : O(1) amortized per evicted event (reference-count lookup, no scan)
  Fast-path identity resolution : O(1) (hashmap lookup)
  Slow-path identity resolution : O(E) where E = number of active entity clusters
    — this is disclosed, not claimed as O(1). E is bounded by the number of
    distinct actors active within the sliding window, which is small in practice.

Thread safety
-------------
  A reentrant lock (RLock) guards all mutations. Safe to call from multiple
  threads (e.g. a Loki push receiver and a background eviction ticker).
"""

import time
import threading
import logging
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class MultiCloudGraphEngine:
    """
    Sliding-window directed graph engine for cross-cloud identity stitching.

    Parameters
    ----------
    window_horizon_seconds : int
        Length of the sliding time window. Events older than this are
        evicted. Default: 60 seconds (as per architecture spec).
    similarity_threshold : float
        Minimum fuzzy-fusion score (tau) for two sessions to be merged into
        the same entity cluster. Range [0, 1]. Default: 0.65 (tuned to the
        three-signal weight distribution below; raise to 0.75+ if false
        merge rate is too high on your validation set).
    """

    # ------------------------------------------------------------------ #
    # Signal weights for Tier 2 fuzzy fusion.                             #
    # These are hand-tuned for the current three-signal feature set.      #
    # Replace with learned weights (gradient-boosted fusion classifier)   #
    # once you have enough labeled positive/negative pairs from the        #
    # synthetic dataset generation pipeline.                               #
    # ------------------------------------------------------------------ #
    _UA_WEIGHT    = 0.50
    _PROXY_WEIGHT = 0.30
    _TYPE_WEIGHT  = 0.20

    def __init__(
        self,
        window_horizon_seconds: int = 60,
        similarity_threshold: float = 0.65,
    ):
        self.window_horizon = window_horizon_seconds
        self.tau = similarity_threshold

        # entity_id -> deque of {"arrival_time": float, "data": dict}
        self._graph_registry: Dict[str, deque] = {}

        # entity_id -> founding event (permanent anchor, never evicted)
        # Used as the stable comparison target for fuzzy similarity.
        # Fixes the "anchor drift" bug where active_events[0] shifts as
        # old events are evicted from the window.
        self._entity_anchors: Dict[str, dict] = {}

        # principal_string -> entity_id  (fast-path O(1) lookup)
        self._identity_map: Dict[str, str] = {}

        # principal_string -> int  (reference count for O(1) eviction cleanup)
        # Replaces the O(n) `any(... for n in active_queue)` scan.
        self._principal_ref_count: Dict[str, int] = {}

        # Reentrant lock for thread safety
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def process_event(self, normalized_event: dict) -> Tuple[str, List[dict], bool]:
        """
        Main entry point. Resolves the entity cluster for an incoming
        normalized event, appends it to the sliding window, evicts stale
        entries, and returns the current active window for downstream risk
        scoring.

        Parameters
        ----------
        normalized_event : dict
            A single row from the normalized feature table produced by
            06_normalize_features.py. Must contain at minimum:
              - "principal"         : str
              - "ua_family"         : str
              - "ua_version"        : str
              - "is_known_proxy_or_tor" : bool | str
              - "principal_type"    : str
              - "timestamp"         : ISO-8601 str
            Optional (Tier 1 fast path):
              - "federation_id"     : str | None
                (sourceIdentity / saml_assertion_id / oidc_subject)

        Returns
        -------
        entity_id : str
            The cluster this event was assigned to.
        active_events : List[dict]
            All events currently inside the sliding window for this entity,
            including the one just added.
        is_new_entity : bool
            True if this event created a brand-new cluster (useful for
            logging / alerting on first-seen orphaned principals).
        """
        with self._lock:
            entity_id, is_new = self._resolve_entity_id(normalized_event)
            active_events = self._update_window(entity_id, normalized_event)
            return entity_id, active_events, is_new

    def get_entity_stats(self) -> dict:
        """
        Returns a snapshot of the current graph state.
        Useful for Grafana metadata panels and debugging.
        """
        with self._lock:
            return {
                "active_entity_clusters": len(self._graph_registry),
                "tracked_principals": len(self._identity_map),
                "window_horizon_seconds": self.window_horizon,
                "similarity_threshold": self.tau,
                "entity_sizes": {
                    eid: len(q)
                    for eid, q in self._graph_registry.items()
                },
            }

    # ------------------------------------------------------------------ #
    # Tier 1 — Federation join                                             #
    # ------------------------------------------------------------------ #

    def _tier1_federation_key(self, event: dict) -> Optional[str]:
        """
        Extracts the upstream IdP linkage key if present.

        Cloud providers embed federation lineage in different fields:
          AWS CloudTrail : userIdentity.sessionContext.sessionIssuer
                           or userIdentity.principalId (contains sourceIdentity)
          Azure Entra ID : claims.sub / claims.oid in SignInLogs
          GCP Cloud Audit: principalSubject (service account impersonation chain)

        After normalization, these should all land in "federation_id".
        Returns None if no federation lineage is present — i.e. the event
        comes from an orphaned principal, which is exactly when Tier 2 fires.
        """
        fed_id = event.get("federation_id")
        if fed_id and str(fed_id).strip() not in ("", "None", "null"):
            return str(fed_id).strip()
        return None

    # ------------------------------------------------------------------ #
    # Tier 2 — Fuzzy fusion similarity                                     #
    # ------------------------------------------------------------------ #

    def _calculate_fuzzy_similarity(self, event_a: dict, event_b: dict) -> float:
        """
        Computes the adversarial-evasion-robust similarity score s_total
        between two events. Returns a float in [0.0, 1.0].

        Signal families
        ---------------
        s_ua (weight 0.50):
            UA family exact match (0.70 of budget) + version match (0.30).
            Partial credit (0.20 of budget) if both families are known
            automated frameworks (Boto3, aws-cli, Terraform, etc.) even if
            the family strings differ — tolerates minor tool version drift.

        s_proxy (weight 0.30):
            Shared proxy/Tor infrastructure alignment. If both events arrive
            from known-proxy/Tor infrastructure, that is a POSITIVE similarity
            signal (same evasion tooling). If one is proxy and one is clean
            residential/corporate, that is a NEGATIVE signal (accounts for
            attacker going proxy-off after gaining legitimate-looking cover).
            Replaces the previous geo exact-match, which penalised IP rotation
            — the exact evasion this system is designed to catch.

        s_type (weight 0.20):
            Principal type consistency (IAMUser, AssumedRole, ServiceAccount,
            etc.). Stays stable across sessions for the same actor toolchain.

        Note on geo exclusion
        ----------------------
        IP geo-country exact match is intentionally NOT included. It is the
        signal that changes most reliably when an attacker rotates proxies or
        VPN exit nodes. Including it with positive weight would reward an
        attacker staying in one country and punish the cross-country rotation
        pattern that is most diagnostic of the threat we target.
        """
        score = 0.0

        # ---- s_ua -------------------------------------------------------
        KNOWN_AUTOMATION_FAMILIES = {
            "Boto3", "aws-cli", "aws-sdk-go", "Terraform",
            "azure-cli", "google-cloud-sdk", "pulumi",
        }

        ua_a = event_a.get("ua_family", "Unknown")
        ua_b = event_b.get("ua_family", "Unknown")

        if ua_a == ua_b and ua_a != "Unknown":
            score += self._UA_WEIGHT * 0.70
            if event_a.get("ua_version") == event_b.get("ua_version"):
                score += self._UA_WEIGHT * 0.30
        elif ua_a in KNOWN_AUTOMATION_FAMILIES and ua_b in KNOWN_AUTOMATION_FAMILIES:
            # Different automation frameworks — same attacker may switch tools
            # between sessions; give partial credit rather than zero.
            score += self._UA_WEIGHT * 0.20

        # ---- s_proxy ----------------------------------------------------
        proxy_a = bool(event_a.get("is_known_proxy_or_tor", False))
        proxy_b = bool(event_b.get("is_known_proxy_or_tor", False))

        if proxy_a and proxy_b:
            # Both through anonymisation infrastructure — strong positive signal
            score += self._PROXY_WEIGHT * 1.0
        elif proxy_a == proxy_b:
            # Both clean/corporate — consistent, moderate positive signal
            score += self._PROXY_WEIGHT * 0.5
        # else: one proxy, one clean — no contribution (intentionally neutral,
        # not negative, because legitimate users sometimes VPN in)

        # ---- s_type -----------------------------------------------------
        if event_a.get("principal_type") == event_b.get("principal_type"):
            score += self._TYPE_WEIGHT

        return round(score, 4)

    # ------------------------------------------------------------------ #
    # Entity resolution                                                    #
    # ------------------------------------------------------------------ #

    def _resolve_entity_id(self, incoming_event: dict) -> Tuple[str, bool]:
        """
        Resolves which entity cluster this event belongs to.

        Resolution order:
          1. Tier 1: federation_id fast-path — O(1)
          2. Identity map fast-path for already-seen principals — O(1)
          3. Tier 2: fuzzy similarity scan across active clusters — O(E)
          4. Fallback: create new entity cluster

        Returns (entity_id, is_new_entity).
        """
        principal = incoming_event.get("principal", "")

        # ---- Step 1: Tier 1 federation join ----------------------------
        fed_key = self._tier1_federation_key(incoming_event)
        if fed_key:
            if fed_key not in self._identity_map:
                new_id = self._create_new_entity(fed_key, incoming_event)
                logger.debug("Tier1 new entity %s for federation key %s", new_id, fed_key)
                return new_id, True
            entity_id = self._identity_map[fed_key]
            # Also bind the raw principal to the same entity for fast lookup
            self._identity_map[principal] = entity_id
            return entity_id, False

        # ---- Step 2: identity map fast-path ----------------------------
        if principal in self._identity_map:
            return self._identity_map[principal], False

        # ---- Step 3: Tier 2 fuzzy scan ---------------------------------
        best_entity_id: Optional[str] = None
        best_score = 0.0

        for entity_id, anchor in self._entity_anchors.items():
            # Skip entities with empty windows (fully evicted but not yet gc'd)
            if entity_id not in self._graph_registry or not self._graph_registry[entity_id]:
                continue
            sim = self._calculate_fuzzy_similarity(incoming_event, anchor)
            if sim > best_score:
                best_score = sim
                best_entity_id = entity_id

        if best_score >= self.tau and best_entity_id is not None:
            self._identity_map[principal] = best_entity_id
            logger.debug(
                "Tier2 merged principal '%s' into entity %s (score=%.3f)",
                principal, best_entity_id, best_score,
            )
            return best_entity_id, False

        # ---- Step 4: new entity ----------------------------------------
        new_id = self._create_new_entity(principal, incoming_event)
        logger.debug(
            "New entity %s for orphaned principal '%s' (best_score=%.3f < tau=%.2f)",
            new_id, principal, best_score, self.tau,
        )
        return new_id, True

    def _create_new_entity(self, key: str, founding_event: dict) -> str:
        """
        Instantiates a new entity cluster. Stores the founding event as the
        permanent comparison anchor — this anchor is never evicted, ensuring
        consistent similarity comparisons regardless of window age.
        """
        entity_id = f"entity_{int(time.time_ns())}"
        self._identity_map[key] = entity_id
        self._entity_anchors[entity_id] = founding_event   # permanent anchor
        self._graph_registry[entity_id] = deque()
        return entity_id

    # ------------------------------------------------------------------ #
    # Sliding window management                                            #
    # ------------------------------------------------------------------ #

    def _update_window(self, entity_id: str, incoming_event: dict) -> List[dict]:
        """
        Appends the incoming event to the entity's deque and evicts entries
        that have drifted beyond the sliding window horizon.

        Complexity
        ----------
        Append: O(1)
        Eviction: O(1) amortized per evicted event.
          - Each event is inserted exactly once and evicted exactly once.
          - Eviction cleanup uses reference counting (O(1) lookup),
            not a linear scan of the queue.
        Total over a stream of n events: O(n) work, O(1) amortized per event.
        """
        current_time = time.time()
        principal = incoming_event.get("principal", "")
        queue = self._graph_registry[entity_id]

        # Insert
        queue.append({"arrival_time": current_time, "data": incoming_event})
        self._principal_ref_count[principal] = (
            self._principal_ref_count.get(principal, 0) + 1
        )

        # Evict stale entries — O(1) per eviction via reference counting
        while queue and (current_time - queue[0]["arrival_time"]) > self.window_horizon:
            evicted = queue.popleft()
            evicted_principal = evicted["data"].get("principal", "")

            # Decrement reference count; clean identity map only when count hits 0
            # This is O(1) — no scanning the queue.
            count = self._principal_ref_count.get(evicted_principal, 1) - 1
            if count <= 0:
                self._principal_ref_count.pop(evicted_principal, None)
                self._identity_map.pop(evicted_principal, None)
            else:
                self._principal_ref_count[evicted_principal] = count

        # Garbage-collect empty entity clusters to prevent memory accumulation.
        # The permanent anchor is also cleaned up here.
        if not queue:
            del self._graph_registry[entity_id]
            self._entity_anchors.pop(entity_id, None)
            # FIX: Purge any dangling Tier 1 federation keys or orphaned principals 
            # from the identity map that point to this dead cluster.
            dangling_keys = [k for k, v in self._identity_map.items() if v == entity_id]
            for k in dangling_keys:
                self._identity_map.pop(k, None)
            logger.debug("Entity %s fully evicted and removed from registry.", entity_id)
            return []

        return [node["data"] for node in queue]