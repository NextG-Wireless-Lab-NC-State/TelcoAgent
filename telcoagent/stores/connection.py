"""Neo4j driver connection + shared helpers.

Provides the ``Neo4jConnection`` driver wrapper plus the sanitisation /
condition-label utilities used by all three store subclasses.

``Neo4jConnection.__init__`` calls ``verify_connectivity()``, so constructing
it RAISES if Neo4j is unreachable. This class is used only by the Explainer
and offline KG construction; the TSFM predictor (``TelcoAgent.predict``)
operates without a live Neo4j instance and never instantiates this class on
the forward-pass path.
"""

import atexit
import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Condition labels for active-condition detection
_KPI_CONDITION_NAMES: Dict[str, str] = {
    "PRB_Util": "congestion",
    "DL_CQI": "poor_channel",
    "DL_iBler": "high_bler",
    "DL_rBler": "high_residual_bler",
    "RRC_Drop_Rate": "high_drop_rate",
    "E2E_Latency": "high_latency",
}


def _sanitize_neo4j_label(raw: str) -> str:
    """Convert an arbitrary string to a valid Neo4j relationship/node label.

    Rules: alphanumeric + underscore only, no leading digit, no empty string.
    Examples: 'calculated_by' → 'CALCULATED_BY', 'has-unit' → 'HAS_UNIT',
              'may report' → 'MAY_REPORT'.
    """
    safe = re.sub(r"[^A-Za-z0-9_]", "_", raw.upper())
    safe = re.sub(r"_+", "_", safe).strip("_")
    if safe and safe[0].isdigit():
        safe = "N_" + safe
    return safe or "RELATED_TO"


# ═══════════════════════════════════════════════════════════════════════════
# Neo4jConnection
# ═══════════════════════════════════════════════════════════════════════════


class Neo4jConnection:
    """Manage a Neo4j driver from env vars or explicit parameters."""

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        from neo4j import GraphDatabase

        self._uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self._user = user or os.environ.get("NEO4J_USER", "neo4j")
        self._password = password or os.environ.get("NEO4J_PASSWORD", "neo4j123")

        self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))
        self._driver.verify_connectivity()
        logger.info("Neo4j connected: %s", self._uri)

        atexit.register(self.close)

    # -- public helpers -------------------------------------------------

    def get_driver(self):
        return self._driver

    def close(self):
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def run(self, cypher: str, **params) -> List[Dict[str, Any]]:
        """Execute a read/write Cypher query and return list of record dicts."""
        with self._driver.session() as session:
            result = session.run(cypher, **params)
            return [record.data() for record in result]

    def ensure_indexes(self) -> None:
        """Create vector indexes + uniqueness constraints if missing."""
        stmts = [
            # Vector indexes
            (
                "CREATE VECTOR INDEX pattern_embedding IF NOT EXISTS "
                "FOR (p:Pattern) ON (p.embedding) "
                "OPTIONS {indexConfig: {`vector.dimensions`: 66, "
                "`vector.similarity_function`: 'cosine'}}"
            ),
            (
                "CREATE VECTOR INDEX cluster_centroid IF NOT EXISTS "
                "FOR (cl:Cluster) ON (cl.centroid) "
                "OPTIONS {indexConfig: {`vector.dimensions`: 66, "
                "`vector.similarity_function`: 'cosine'}}"
            ),
            # Uniqueness constraints
            (
                "CREATE CONSTRAINT pattern_id_unique IF NOT EXISTS "
                "FOR (p:Pattern) REQUIRE p.pattern_id IS UNIQUE"
            ),
            (
                "CREATE CONSTRAINT cluster_id_unique IF NOT EXISTS "
                "FOR (cl:Cluster) REQUIRE cl.cluster_id IS UNIQUE"
            ),
            # Ontology node constraints (same as load_kg_neo4j.py)
            ("CREATE CONSTRAINT IF NOT EXISTS " "FOR (k:KPI) REQUIRE k.name IS UNIQUE"),
            ("CREATE CONSTRAINT IF NOT EXISTS " "FOR (c:CausalChain) REQUIRE c.name IS UNIQUE"),
            ("CREATE CONSTRAINT IF NOT EXISTS " "FOR (cat:Category) REQUIRE cat.name IS UNIQUE"),
            # KG node type indexes (from enriched_kg.json push)
            "CREATE INDEX IF NOT EXISTS FOR (n:KPIDefinition) ON (n.name)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.name)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Measurement) ON (n.name)",
        ]
        with self._driver.session() as session:
            for stmt in stmts:
                try:
                    session.run(stmt)
                except Exception as exc:
                    # Some constraints may fail on Community edition — log and continue
                    logger.debug("Index/constraint skipped: %s", exc)

        logger.info("Neo4j indexes and constraints ensured")
