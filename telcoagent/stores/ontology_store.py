"""Neo4jOntologyStore — Cypher-based 3GPP ontology graph.

Pushes the ``ThreeGPPOntology`` dataclass bundle into Neo4j and serves
condition-aware retrieval queries to the ReAct agents.  Internal helper
views ``_Neo4jGraphReadThrough`` and ``_NodeView`` are defined here to
avoid cross-module coupling.
"""

import json
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from telcoagent.config import CORE_KPI_NAMES
from telcoagent.ontology.core import ThreeGPPOntology

from .connection import _KPI_CONDITION_NAMES, Neo4jConnection, _sanitize_neo4j_label

logger = logging.getLogger(__name__)


class _Neo4jGraphReadThrough:
    """Minimal duck-type wrapper for ``graph_rag.graph`` access.

    ``query_extracted_knowledge`` in react_tools.py accesses
    ``graph_rag.graph.nodes`` and ``.edges(data=True)``.  This wrapper
    issues Cypher reads instead of holding a NetworkX graph in memory.
    """

    def __init__(self, conn: Neo4jConnection):
        self._conn = conn

    def _node_dict(self) -> "_NodeView":
        return _NodeView(self._conn)

    @property
    def nodes(self):
        return self._node_dict()

    def edges(self, data: bool = False):
        records = self._conn.run(
            "MATCH (a)-[r]->(b) "
            "RETURN a.name AS source, b.name AS target, properties(r) AS props"
        )
        if data:
            return [(rec["source"], rec["target"], rec["props"]) for rec in records]
        return [(rec["source"], rec["target"]) for rec in records]


class _NodeView:
    """Minimal duck-type for ``G.nodes[name]`` attribute access."""

    def __init__(self, conn: Neo4jConnection):
        self._conn = conn

    def __contains__(self, name: str) -> bool:
        recs = self._conn.run("MATCH (n {name: $name}) RETURN n LIMIT 1", name=name)
        return len(recs) > 0

    def __getitem__(self, name: str) -> Dict[str, Any]:
        recs = self._conn.run(
            "MATCH (n {name: $name}) RETURN properties(n) AS props LIMIT 1",
            name=name,
        )
        if not recs:
            raise KeyError(name)
        return recs[0]["props"]

    def __iter__(self):
        recs = self._conn.run("MATCH (n) RETURN n.name AS name")
        return iter(r["name"] for r in recs)

    def __call__(self, data: bool = False):
        recs = self._conn.run("MATCH (n) RETURN n.name AS name, properties(n) AS props")
        if data:
            return [(r["name"], r["props"]) for r in recs]
        return [r["name"] for r in recs]


class Neo4jOntologyStore:
    """Cypher-based 3GPP ontology GraphRAG with condition-aware retrieval.

    Public API:
    - retrieve(target_kpis, prediction_stats, iteration_history) -> str
    - merge_extracted_kg(kg_graph) -> dict
    - graph: read-through property for react_tools query_extracted_knowledge
    """

    def __init__(
        self,
        conn: Optional[Neo4jConnection] = None,
        ontology: Optional[ThreeGPPOntology] = None,
        max_relationships: int = 8,
        max_chains: int = 3,
        ego_radius: int = 2,
        kg_path: Optional[str] = None,
    ):
        self._conn = conn or Neo4jConnection()
        if ontology is None:
            raise ValueError(
                "Neo4jOntologyStore requires an explicit ontology; "
                "pass get_default_ontology() at the call site."
            )
        self.ontology = ontology
        self.max_relationships = max_relationships
        self.max_chains = max_chains
        self.ego_radius = ego_radius

        # Seed the ontology data into Neo4j
        self._seed_ontology()

        # Seed physics/environment/formula/CQI nodes from pre-built KG JSON
        if kg_path:
            self._seed_from_kg_json(kg_path)

        # Read-through wrapper for react_tools compatibility
        self._graph_wrapper = _Neo4jGraphReadThrough(self._conn)

        logger.info(
            "Neo4jOntologyStore initialised (max_rels=%d, max_chains=%d, radius=%d)",
            max_relationships,
            max_chains,
            ego_radius,
        )

    @property
    def graph(self):
        """Duck-type for ``graph_rag.graph`` used by react_tools."""
        return self._graph_wrapper

    # -- seed ontology into Neo4j --------------------------------------

    def _seed_ontology(self) -> None:
        """MERGE ontology KPIs, relationships, causal chains into Neo4j.

        Only fields the spec-extracted KG actually populates are written.
        Hand-curated category, kpi_type, min/max, and threshold fields are
        intentionally absent.
        """
        with self._conn.get_driver().session() as session:
            # KPI nodes — push only KG-populated attributes
            for name, kpi in self.ontology.kpis.items():
                session.run(
                    """
                    MERGE (k:KPI {name: $name})
                    SET k.description = $description,
                        k.formula = $formula,
                        k.unit = $unit,
                        k.source_specs = $source_specs,
                        k.source_clauses = $source_clauses,
                        k.confidence = $confidence
                    """,
                    name=name,
                    description=kpi.description,
                    formula=kpi.formula,
                    unit=kpi.unit,
                    source_specs=list(kpi.source_specs),
                    source_clauses=list(kpi.source_clauses),
                    confidence=kpi.confidence,
                )

            # Category seeding has been removed — the KG does not currently
            # emit a category attribute for KPIDefinition nodes.

            # Relationships (AFFECTS edges)
            for rel in self.ontology.relationships:
                for ep in (rel.source_kpi, rel.target_kpi):
                    session.run("MERGE (k:KPI {name: $name})", name=ep)

                session.run(
                    """
                    MATCH (s:KPI {name: $src}), (t:KPI {name: $tgt})
                    MERGE (s)-[r:AFFECTS]->(t)
                    SET r.strength = $strength,
                        r.condition = $condition,
                        r.reason = $reason,
                        r.relation_type = $rel_type
                    """,
                    src=rel.source_kpi,
                    tgt=rel.target_kpi,
                    strength=rel.strength,
                    condition=rel.condition,
                    reason=rel.reason,
                    rel_type=rel.relation_type,
                )

            # Causal chains
            for chain in self.ontology.causal_chains:
                session.run(
                    """
                    MERGE (cc:CausalChain {name: $name})
                    SET cc.description = $desc,
                        cc.steps = $steps,
                        cc.trigger_condition = $trigger
                    """,
                    name=chain.name,
                    desc=chain.description,
                    steps=chain.steps,
                    trigger=chain.trigger_condition,
                )
                # Link to involved KPIs
                for step in chain.steps:
                    for kpi_name in self.ontology.kpis:
                        if kpi_name in step:
                            session.run(
                                """
                                MATCH (cc:CausalChain {name: $chain}),
                                      (k:KPI {name: $kpi})
                                MERGE (cc)-[:INVOLVES]->(k)
                                """,
                                chain=chain.name,
                                kpi=kpi_name,
                            )

        logger.info("Ontology seeded into Neo4j")

    # -- physics/environment KG seeding --------------------------------

    def _seed_from_kg_json(self, kg_path: str) -> None:
        """Load EnvironmentType, Formula, PhysicalQuantity, CQIEntry nodes from
        the pre-built enriched_kg.json into Neo4j."""
        import json as _json

        try:
            with open(kg_path, "r") as f:
                kg_data = _json.load(f)
        except Exception as exc:
            logger.warning("Could not load KG JSON from %s: %s", kg_path, exc)
            return

        nodes = kg_data.get("nodes", [])
        edges = kg_data.get("links", kg_data.get("edges", []))

        # Map node id → attrs for edge resolution
        node_map = {n["id"]: n for n in nodes}

        with self._conn.get_driver().session() as session:
            for n in nodes:
                ntype = n.get("node_type", "")
                nid = n["id"]
                props = {k: v for k, v in n.items() if k not in ("id", "node_type")}
                props["name"] = nid

                if ntype == "EnvironmentType":
                    session.run(
                        """
                        MERGE (e:EnvironmentType {name: $name})
                        SET e += $props
                        """,
                        name=nid,
                        props=props,
                    )
                elif ntype == "Formula":
                    session.run(
                        """
                        MERGE (f:Formula {name: $name})
                        SET f += $props
                        """,
                        name=nid,
                        props=props,
                    )
                elif ntype == "PhysicalQuantity":
                    session.run(
                        """
                        MERGE (p:PhysicalQuantity {name: $name})
                        SET p += $props
                        """,
                        name=nid,
                        props=props,
                    )
                elif ntype == "CQIEntry":
                    session.run(
                        """
                        MERGE (c:CQIEntry {name: $name})
                        SET c += $props
                        """,
                        name=nid,
                        props=props,
                    )
                # KPIDefinition and CausalChain already seeded by _seed_ontology

            for e in edges:
                src = e.get("source", e.get("from", ""))
                tgt = e.get("target", e.get("to", ""))
                rel = e.get("relation_type", e.get("type", "RELATED_TO"))
                edge_props = {
                    k: v
                    for k, v in e.items()
                    if k not in ("source", "target", "from", "to", "relation_type", "type")
                }

                src_type = node_map.get(src, {}).get("node_type", "")
                tgt_type = node_map.get(tgt, {}).get("node_type", "")

                # Only create edges involving physics node types
                physics_types = {"EnvironmentType", "Formula", "PhysicalQuantity", "CQIEntry"}
                if src_type in physics_types or tgt_type in physics_types:
                    safe_rel = rel.upper().replace(" ", "_").replace("-", "_")
                    try:
                        session.run(
                            f"""
                            MATCH (a {{name: $src}}), (b {{name: $tgt}})
                            MERGE (a)-[r:`{safe_rel}`]->(b)
                            SET r += $props
                            """,
                            src=src,
                            tgt=tgt,
                            props=edge_props,
                        )
                    except Exception as exc:
                        logger.debug("Edge %s->%s skipped: %s", src, tgt, exc)

        logger.info("Physics/environment KG seeded from %s", kg_path)

    # -- physics/environment queries ------------------------------------

    def query_physics_context(self, environment_type: str) -> str:
        """Return 3GPP physics context (path loss models, formulas) for an
        environment type (e.g. 'UMa', 'UMi', 'RMa', 'InH')."""
        env_upper = environment_type.upper()

        # Match EnvironmentType nodes whose name contains the prefix
        records = self._conn.run(
            """
            MATCH (e:EnvironmentType)
            WHERE toUpper(e.name) STARTS WITH $prefix
            OPTIONAL MATCH (e)-[r]->(f:Formula)
            RETURN e, collect({formula: f, rel: type(r)}) AS formulas
            """,
            prefix=env_upper,
        )

        if not records:
            return f"No physics data found for environment type '{environment_type}'."

        parts = [f"[3GPP Physics — {environment_type}]"]
        for row in records:
            env = row["e"]
            env_name = env.get("name", "?")
            scenario = env.get("scenario", env.get("description", ""))
            parts.append(f"\nEnvironment: {env_name}")
            if scenario:
                parts.append(f"  Scenario: {scenario}")
            for item in row.get("formulas", []):
                f = item.get("formula")
                if f:
                    f_name = f.get("name", "")
                    f_expr = f.get("formula_expression", f.get("expression", ""))
                    f_desc = f.get("description", "")
                    parts.append(f"  Formula [{f_name}]: {f_expr}")
                    if f_desc:
                        parts.append(f"    → {f_desc}")

        # Also get path loss model parameters if stored
        pl_records = self._conn.run(
            """
            MATCH (e:EnvironmentType)
            WHERE toUpper(e.name) STARTS WITH $prefix
            RETURN e.name AS name,
                   e.path_loss_A AS A, e.path_loss_B AS B, e.path_loss_C AS C,
                   e.shadow_fading_std AS sigma, e.los_condition AS los
            """,
            prefix=env_upper,
        )
        for row in pl_records:
            if row.get("A") is not None:
                name = row["name"]
                A, B, C = row.get("A"), row.get("B"), row.get("C")
                sigma = row.get("sigma")
                los = row.get("los", "")
                parts.append(
                    f"\nPath Loss [{name}]: PL = {A}·log10(d) + {B} + {C}·log10(fc)"
                    f"  (shadow_fading_std={sigma} dB, LOS={los})"
                )

        return "\n".join(parts)

    def query_cqi_context(self, cqi_value: Optional[int] = None) -> str:
        """Return CQI table info (TS 38.214 Table 5.2.2.1-2).

        If cqi_value is given, returns info for that specific index.
        Otherwise returns the full table summary.
        """
        if cqi_value is not None:
            records = self._conn.run(
                """
                MATCH (c:CQIEntry)
                WHERE c.cqi_index = $idx OR c.name ENDS WITH $idx_str
                RETURN c
                LIMIT 1
                """,
                idx=int(cqi_value),
                idx_str=str(cqi_value),
            )
            if not records:
                return f"CQI {cqi_value}: not found in table."
            c = records[0]["c"]
            return (
                f"CQI {cqi_value}: modulation={c.get('modulation','?')}, "
                f"code_rate={c.get('code_rate','?')}, "
                f"spectral_efficiency={c.get('spectral_efficiency','?')} bits/s/Hz, "
                f"SINR_threshold≈{c.get('sinr_threshold_db','?')} dB  [TS 38.214 Table 5.2.2.1-2]"
            )
        else:
            records = self._conn.run("""
                MATCH (c:CQIEntry)
                RETURN c ORDER BY c.cqi_index
                """)
            if not records:
                return "CQI table not loaded into Neo4j."
            lines = ["CQI Table (TS 38.214 Table 5.2.2.1-2):"]
            for row in records:
                c = row["c"]
                lines.append(
                    f"  CQI {c.get('cqi_index','?'):>2}: {c.get('modulation','?'):>8}  "
                    f"CR={c.get('code_rate','?')}  η={c.get('spectral_efficiency','?')}  "
                    f"SINR≥{c.get('sinr_threshold_db','?')} dB"
                )
            return "\n".join(lines)

    # -- condition detection -------------------------------------------

    def _detect_active_conditions(
        self,
        prediction_stats: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, str]:
        """Active-condition detection requires per-KPI thresholds + good/bad
        direction.  Both fields were hand-curated and have been archived,
        and the KG does not currently emit them.  This method now returns
        ``{}`` until the KG starts populating those attributes (or an
        equivalent KG-native rule mechanism replaces it).
        """
        return {}

    # -- subgraph extraction (Cypher 1..2 hop) -------------------------

    def _extract_subgraph(
        self,
        target_kpis: Optional[List[str]] = None,
        active_condition_keys: Optional[Set[str]] = None,
        iteration_history: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        target_kpis = target_kpis or list(self.ontology.kpis.keys())
        active_condition_keys = active_condition_keys or set()
        kpi_weights: Dict[str, float] = {}
        if iteration_history:
            kpi_weights = iteration_history.get("kpi_weights", {})

        # Cypher: 1..ego_radius hops from target KPIs
        records = self._conn.run(
            """
            UNWIND $targets AS tgt
            MATCH (s:KPI {name: tgt})-[*1..2]-(n:KPI)
            WITH COLLECT(DISTINCT n.name) + $targets AS neighbourhood
            UNWIND neighbourhood AS nb
            WITH COLLECT(DISTINCT nb) AS nbSet
            MATCH (u:KPI)-[r:AFFECTS]->(v:KPI)
            WHERE u.name IN nbSet OR v.name IN nbSet
            RETURN u.name AS source, v.name AS target,
                   r.relation_type AS relation_type,
                   r.strength AS strength,
                   r.condition AS condition,
                   r.reason AS reason
            """,
            targets=target_kpis,
        )

        kpi_to_condition = {
            kpi: cond for kpi, cond in _KPI_CONDITION_NAMES.items() if cond in active_condition_keys
        }

        scored_edges: List[Dict[str, Any]] = []
        for rec in records:
            score = 0.0
            strength = rec.get("strength", "moderate")
            score += {"critical": 4.0, "strong": 3.0, "moderate": 2.0, "weak": 1.0}.get(
                strength, 1.0
            )

            if rec["source"] in target_kpis or rec["target"] in target_kpis:
                score += 2.0

            condition = rec.get("condition") or ""
            if condition:
                for kpi_name in kpi_to_condition:
                    if kpi_name in condition:
                        score += 3.0

            for endpoint in (rec["source"], rec["target"]):
                weight = kpi_weights.get(endpoint, 1.0)
                if weight > 1.0:
                    score *= weight

            scored_edges.append(
                {
                    "source": rec["source"],
                    "target": rec["target"],
                    "relation_type": rec.get("relation_type", "RELATES_TO"),
                    "strength": strength,
                    "condition": condition,
                    "reason": rec.get("reason", ""),
                    "score": score,
                }
            )

        scored_edges.sort(key=lambda e: e["score"], reverse=True)
        return scored_edges

    # -- causal chain selection ----------------------------------------

    def _select_causal_chains(
        self,
        target_kpis: Optional[List[str]] = None,
        active_condition_keys: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        target_kpis = target_kpis or list(self.ontology.kpis.keys())
        active_condition_keys = active_condition_keys or set()

        active_kpis = {
            kpi for kpi, cond in _KPI_CONDITION_NAMES.items() if cond in active_condition_keys
        }

        records = self._conn.run("""
            MATCH (cc:CausalChain)
            RETURN cc.name AS name,
                   cc.description AS description,
                   cc.steps AS steps,
                   cc.trigger_condition AS trigger
            """)

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for rec in records:
            score = 0.0
            is_active = False
            trigger = rec.get("trigger") or ""
            if trigger:
                for kpi_name in active_kpis:
                    if kpi_name in trigger:
                        is_active = True
                        score += 5.0

            steps = rec.get("steps") or []
            for step in steps:
                for kpi in target_kpis:
                    if kpi in step:
                        score += 1.0

            scored.append(
                (
                    score,
                    {
                        "name": rec["name"],
                        "description": rec["description"],
                        "steps": steps,
                        "trigger": trigger,
                        "active": is_active,
                    },
                )
            )

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[: self.max_chains]]

    # -- formatting -----------------------------------------------------

    def _format_subgraph(
        self,
        active_conditions: Dict[str, str],
        target_kpis: List[str],
        scored_edges: List[Dict[str, Any]],
        chains: List[Dict[str, Any]],
    ) -> str:
        lines: List[str] = []

        if active_conditions:
            labels = list(active_conditions.values())
            lines.append(f"[ACTIVE CONDITIONS] {', '.join(labels)}")
        else:
            lines.append("[ACTIVE CONDITIONS] None detected")
        lines.append("")

        lines.append("[KPI METADATA]")
        for kpi_name in target_kpis:
            kpi = self.ontology.get_kpi(kpi_name)
            if kpi is None:
                continue
            parts = [f"  {kpi_name}:"]
            if kpi.unit:
                parts.append(f"unit={kpi.unit}")
            if kpi.formula:
                parts.append(f"formula={kpi.formula}")
            if kpi.source_specs:
                parts.append(f"spec={', '.join(kpi.source_specs)}")
            lines.append(" ".join(parts))
        lines.append("")

        top_edges = scored_edges[: self.max_relationships]
        if top_edges:
            lines.append("[ACTIVE RELATIONSHIPS]")
            for edge in top_edges:
                cond = f" IF {edge['condition']}" if edge["condition"] else ""
                reason = f" [{edge['reason']}]" if edge.get("reason") else ""
                lines.append(
                    f"  {edge['source']} --{edge['relation_type']}({edge['strength']})--> "
                    f"{edge['target']}{cond}{reason}"
                )
            lines.append("")

        if chains:
            lines.append("[CAUSAL CHAINS]")
            for ch in chains:
                tag = " [ACTIVE]" if ch["active"] else ""
                steps = " -> ".join(ch["steps"])
                lines.append(f"  {ch['name']}{tag}: {steps}")
            lines.append("")

        return "\n".join(lines)

    # -- merge extracted KG (from TelcoAgent-RAG kg_builder) ----------------

    def merge_extracted_kg(
        self,
        kg_graph,
        min_confidence: float = 0.5,
        min_quality: float = 0.5,
    ) -> Dict[str, int]:
        """Merge NetworkX KG (from kg_builder) into Neo4j via Cypher."""
        stats = {
            "nodes_enriched": 0,
            "nodes_added": 0,
            "edges_enriched": 0,
            "edges_added": 0,
            "nodes_skipped": 0,
            "edges_skipped": 0,
        }

        with self._conn.get_driver().session() as session:
            for node_id, attrs in kg_graph.nodes(data=True):
                confidence = attrs.get("confidence", 0.0)
                quality_scores = attrs.get("quality_scores", [])
                mean_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 1.0
                if confidence < min_confidence or mean_quality < min_quality:
                    stats["nodes_skipped"] += 1
                    continue

                # Check if node exists
                existing = session.run(
                    "MATCH (n {name: $name}) RETURN n LIMIT 1", name=node_id
                ).data()

                props = {
                    "_enriched_from_pdf": True,
                }
                for field in ("formula", "description"):
                    if field in attrs:
                        props[field] = attrs[field]

                if existing:
                    session.run(
                        "MATCH (n {name: $name}) SET n += $props",
                        name=node_id,
                        props=props,
                    )
                    stats["nodes_enriched"] += 1
                else:
                    node_type = attrs.get("node_type", "Entity")
                    props["node_type"] = node_type
                    props["name"] = node_id
                    props["_extracted_node_type"] = node_type
                    session.run(
                        "CREATE (n:KPI $props)",
                        props=props,
                    )
                    stats["nodes_added"] += 1

            for u, v, attrs in kg_graph.edges(data=True):
                confidence = attrs.get("confidence", 0.0)
                quality_score = attrs.get("quality_score", 1.0)
                if confidence < min_confidence or quality_score < min_quality:
                    stats["edges_skipped"] += 1
                    continue

                # Ensure endpoints exist
                for ep in (u, v):
                    session.run("MERGE (n:KPI {name: $name})", name=ep)

                existing_edge = session.run(
                    "MATCH (a {name: $u})-[r]->(b {name: $v}) RETURN r LIMIT 1",
                    u=u,
                    v=v,
                ).data()

                props: Dict[str, Any] = {"_enriched_from_pdf": True}
                for field in ("source_spec", "source_clause", "raw_text", "triple_type"):
                    if field in attrs:
                        props[f"_extracted_{field}"] = attrs[field]

                if existing_edge:
                    session.run(
                        "MATCH (a {name: $u})-[r]->(b {name: $v}) SET r += $props",
                        u=u,
                        v=v,
                        props=props,
                    )
                    stats["edges_enriched"] += 1
                else:
                    rel_type = attrs.get("relation_type", "RELATED_TO")
                    props["relation_type"] = rel_type
                    props["strength"] = "weak"
                    props["condition"] = ""
                    props["reason"] = attrs.get("raw_text", "")
                    session.run(
                        """
                        MATCH (a {name: $u}), (b {name: $v})
                        CREATE (a)-[r:AFFECTS $props]->(b)
                        """,
                        u=u,
                        v=v,
                        props=props,
                    )
                    stats["edges_added"] += 1

        logger.info(
            "Merged extracted KG: enriched %d nodes, added %d nodes, "
            "enriched %d edges, added %d edges (skipped %d nodes, %d edges)",
            stats["nodes_enriched"],
            stats["nodes_added"],
            stats["edges_enriched"],
            stats["edges_added"],
            stats["nodes_skipped"],
            stats["edges_skipped"],
        )
        return stats

    # -- full KG JSON push (best-practice, typed relationships) -----------

    def _ensure_kg_indexes(self) -> None:
        """Create lookup indexes for KG-sourced node types if not present."""
        stmts = [
            "CREATE INDEX IF NOT EXISTS FOR (n:KPIDefinition) ON (n.name)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.name)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Measurement) ON (n.name)",
            "CREATE INDEX IF NOT EXISTS FOR (n:CausalChain) ON (n.name)",
        ]
        with self._conn.get_driver().session() as session:
            for stmt in stmts:
                try:
                    session.run(stmt)
                except Exception as exc:
                    logger.debug("KG index skipped: %s", exc)

    def push_kg_json(
        self,
        kg_path: str,
        clear_kg: bool = False,
        min_confidence: float = 0.0,
    ) -> Dict[str, int]:
        """Push enriched_kg.json to Neo4j with typed relationship labels.

        Idempotent — all writes use MERGE.  Each node is labelled with its
        ``node_type`` (KPIDefinition, Entity, Measurement, CausalChain, …).
        Each relationship is typed using the sanitised ``relation_type`` field,
        so semantic edges like INCREASES, DETERMINES, LIMITS appear as first-
        class Neo4j relationship types for proper graph visualisation.

        Batches writes with UNWIND for performance (~2 500 nodes / 2 600 edges
        in the standard enriched_kg.json complete in < 30 s on localhost).

        Args:
            kg_path: Path to enriched_kg.json (NetworkX node-link format).
            clear_kg: Delete all nodes marked ``_from_kg=true`` before
                re-pushing.  Only use during development / sweep reruns.
            min_confidence: Drop ``ontology_relationship`` edges below this
                value.  Measurement and kpi_definition edges are unfiltered.

        Returns:
            Stats dict with keys ``nodes_merged``, ``edges_merged``,
            ``edges_skipped``.
        """
        stats: Dict[str, int] = {
            "nodes_merged": 0,
            "edges_merged": 0,
            "edges_skipped": 0,
        }

        try:
            with open(kg_path) as f:
                kg_data = json.load(f)
        except Exception as exc:
            logger.error("push_kg_json: cannot read %s: %s", kg_path, exc)
            return stats

        nodes = kg_data.get("nodes", [])
        edges = kg_data.get("links", kg_data.get("edges", []))

        if clear_kg:
            with self._conn.get_driver().session() as session:
                session.run("MATCH (n) WHERE n._from_kg = true DETACH DELETE n")
            logger.info("push_kg_json: cleared existing _from_kg nodes")

        self._ensure_kg_indexes()

        # ── Nodes: batch by node_type for correct Cypher label ──────────
        nodes_by_type: Dict[str, list] = defaultdict(list)
        for n in nodes:
            ntype = n.get("node_type") or "Entity"
            nodes_by_type[ntype].append(n)

        with self._conn.get_driver().session() as session:
            for ntype, batch in nodes_by_type.items():
                label = _sanitize_neo4j_label(ntype)
                rows = [
                    {
                        "name": n["id"],
                        "props": {k: v for k, v in n.items() if k != "id" and v is not None}
                        | {"name": n["id"], "_from_kg": True},
                    }
                    for n in batch
                ]
                session.run(
                    f"UNWIND $rows AS row "
                    f"MERGE (n:{label} {{name: row.name}}) "
                    f"SET n += row.props",
                    rows=rows,
                )
                stats["nodes_merged"] += len(batch)
                logger.debug("push_kg_json: merged %d :%s nodes", len(batch), label)

        # ── Edges: filter + batch by sanitised relationship label ────────
        edges_by_label: Dict[str, list] = defaultdict(list)
        for e in edges:
            src = e.get("source", e.get("from", ""))
            tgt = e.get("target", e.get("to", ""))
            if not src or not tgt:
                stats["edges_skipped"] += 1
                continue

            triple_type = e.get("triple_type", "")
            if triple_type in ("ontology_relationship", "relationship"):
                confidence = float(e.get("confidence") or 1.0)
                if confidence < min_confidence:
                    stats["edges_skipped"] += 1
                    continue

            raw_rel = e.get("relation_type") or e.get("type") or "RELATED_TO"
            edges_by_label[_sanitize_neo4j_label(raw_rel)].append(e)

        with self._conn.get_driver().session() as session:
            for rel_label, batch in edges_by_label.items():
                rows = [
                    {
                        "src": e.get("source", e.get("from", "")),
                        "tgt": e.get("target", e.get("to", "")),
                        "props": {
                            k: v
                            for k, v in e.items()
                            if k
                            not in (
                                "source",
                                "target",
                                "from",
                                "to",
                                "relation_type",
                                "type",
                            )
                            and v is not None
                        },
                    }
                    for e in batch
                ]
                try:
                    session.run(
                        f"UNWIND $rows AS row "
                        f"MATCH (a {{name: row.src}}), (b {{name: row.tgt}}) "
                        f"MERGE (a)-[r:{rel_label}]->(b) "
                        f"SET r += row.props",
                        rows=rows,
                    )
                    stats["edges_merged"] += len(batch)
                except Exception as exc:
                    logger.warning(
                        "push_kg_json: batch for :%s failed (%d edges): %s",
                        rel_label,
                        len(batch),
                        exc,
                    )
                    stats["edges_skipped"] += len(batch)

        logger.info(
            "push_kg_json: merged %d nodes, %d edges, skipped %d edges from %s",
            stats["nodes_merged"],
            stats["edges_merged"],
            stats["edges_skipped"],
            kg_path,
        )
        return stats

    # -- public API: retrieve ------------------------------------------

    def retrieve(
        self,
        target_kpis: Optional[List[str]] = None,
        prediction_stats: Optional[Dict[str, Dict[str, Any]]] = None,
        iteration_history: Optional[Dict[str, Any]] = None,
    ) -> str:
        if target_kpis is None:
            target_kpis = list(CORE_KPI_NAMES)

        if prediction_stats:
            normalised: Dict[str, Dict[str, Any]] = {}
            for k, v in prediction_stats.items():
                if isinstance(v, (int, float)):
                    normalised[k] = {"mean": v}
                else:
                    normalised[k] = v
            prediction_stats = normalised

        active_conditions = self._detect_active_conditions(prediction_stats)
        active_keys = set(active_conditions.keys())
        scored_edges = self._extract_subgraph(
            target_kpis,
            active_keys,
            iteration_history,
        )
        chains = self._select_causal_chains(target_kpis, active_keys)

        return self._format_subgraph(active_conditions, target_kpis, scored_edges, chains)


# ═══════════════════════════════════════════════════════════════════════════
# Neo4jClusterStore — replaces PromptBank FAISS retrieval index
# ═══════════════════════════════════════════════════════════════════════════
