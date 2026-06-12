"""MCP-style runtime tools used by the Predictor and Explainer agents.

Two tools live here:

1. :class:`StandardKnowledgeGraphMCP` — KG-driven retrieval over the
   spec-extracted ontology (``ThreeGPPOntology``).  Optionally queries a
   Neo4j store for the same content; there is no hand-curated fallback.
2. :class:`OpenStreetMapMCP` — runtime spatial-context retrieval (POIs,
   road network, environment classification).  Pure external data fetch;
   not knowledge.

Hand-curated validators, the sine/cosine periodicity helper, the
``DataAnalystMCP`` 3-sigma critic, and the ``physics_3gpp`` Python fallback
have been retired and must not be reintroduced.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from ..ontology.core import get_default_ontology


class BaseMCP:
    """Base class for MCP-style runtime tools."""

    def __init__(self, name: str):
        self.name = name


# =============================================================================
# StandardKnowledgeGraphMCP — KG retrieval, no in-Python ontology fallback
# =============================================================================


class StandardKnowledgeGraphMCP(BaseMCP):
    """KG-backed retrieval of KPI definitions, relationships, and causal chains.

    All knowledge comes from :class:`ThreeGPPOntology`, which itself loads
    everything from the enriched KG JSON.  When a Neo4j store is provided
    via the constructor, physics-style queries are routed through it; if
    Neo4j is absent, those queries report unavailable rather than falling
    back to a hand-curated table.
    """

    def __init__(self, neo4j_store=None):
        super().__init__("StandardKnowledgeGraphMCP")
        self._3gpp_ontology = get_default_ontology()
        self._neo4j_store = neo4j_store
        self.ontology = self._build_ontology_dict()

    def _build_ontology_dict(self) -> Dict[str, Any]:
        """Project the typed ontology into a dict for legacy dict-based callers."""
        ont = self._3gpp_ontology

        constraints: Dict[str, Any] = {}
        for kpi in ont.kpis.values():
            entry: Dict[str, Any] = {
                "description": kpi.description,
            }
            if kpi.unit:
                entry["unit"] = kpi.unit
            if kpi.formula:
                entry["formula"] = kpi.formula
            if kpi.source_specs:
                entry["source_specs"] = list(kpi.source_specs)
            if kpi.source_clauses:
                entry["source_clauses"] = list(kpi.source_clauses)
            constraints[kpi.short_name] = entry

        relationships: List[Dict[str, Any]] = []
        for rel in ont.relationships:
            r: Dict[str, Any] = {
                "source": rel.source_kpi,
                "relation": rel.relation_type,
                "target": rel.target_kpi,
                "strength": rel.strength,
                "reason": rel.reason,
            }
            if rel.condition:
                r["condition"] = rel.condition
            if rel.spec_reference:
                r["spec_reference"] = rel.spec_reference
            relationships.append(r)

        causal_chains: List[Dict[str, Any]] = [
            {
                "name": ch.name,
                "chain": ch.steps,
                "description": ch.description,
            }
            for ch in ont.causal_chains
        ]

        return {
            "constraints": constraints,
            "relationships": relationships,
            "causal_chains": causal_chains,
        }

    # -- queries ----------------------------------------------------------

    def query_impact_analysis(self, kpi_name: str) -> List[str]:
        """Return KG-backed lines describing a KPI plus its in/out edges."""
        results: List[str] = []
        ont = self._3gpp_ontology

        kpi_def = ont.get_kpi(kpi_name)
        if kpi_def:
            if kpi_def.description:
                results.append(f"[Definition] {kpi_def.description}")
            if kpi_def.formula:
                results.append(f"[Formula] {kpi_def.formula}")
            if kpi_def.unit:
                results.append(f"[Unit] {kpi_def.unit}")
            if kpi_def.source_specs:
                results.append(
                    f"[Reference] {', '.join(kpi_def.source_specs)}"
                    + (f" §{', '.join(kpi_def.source_clauses)}" if kpi_def.source_clauses else "")
                )

        related = ont.get_related_kpis(kpi_name)
        for rel in related.get("affects", []):
            piece = f"[Relation] {kpi_name} {rel.relation_type} {rel.target_kpi} ({rel.strength})"
            if rel.condition:
                piece += f" when {rel.condition}"
            if rel.reason:
                piece += f" — {rel.reason}"
            results.append(piece)

        for rel in related.get("affected_by", []):
            piece = f"[Relation] {rel.source_kpi} {rel.relation_type} {kpi_name} ({rel.strength})"
            if rel.condition:
                piece += f" when {rel.condition}"
            if rel.reason:
                piece += f" — {rel.reason}"
            results.append(piece)

        for chain in ont.get_causal_chains_for_kpi(kpi_name):
            results.append(
                f"[CausalChain] {chain.name}: {' → '.join(chain.steps)} — {chain.description}"
            )

        return results

    def get_constraints(self, kpi_name: str) -> Dict[str, Any]:
        return self.ontology["constraints"].get(kpi_name, {})

    def get_causal_chains(self) -> List[Dict[str, Any]]:
        return self.ontology["causal_chains"]

    def get_ontology_summary(self) -> str:
        return self._3gpp_ontology.get_ontology_summary()

    def get_kpi_info(self, kpi_name: str) -> str:
        return self._3gpp_ontology.format_kpi_info(kpi_name)

    def query_physics_context(self, environment_type: str) -> str:
        """Physics context query.  Requires Neo4j; no in-Python fallback."""
        if self._neo4j_store is not None:
            return self._neo4j_store.query_physics_context(environment_type)
        return (
            f"Physics context for '{environment_type}' unavailable: "
            "no Neo4j store configured and the in-Python physics table "
            "has been retired."
        )

    def query_cqi_context(self, cqi_value: Optional[int] = None) -> str:
        """CQI table query.  Requires Neo4j; no in-Python fallback."""
        if self._neo4j_store is not None:
            return self._neo4j_store.query_cqi_context(cqi_value)
        return (
            "CQI table unavailable: no Neo4j store configured and the "
            "in-Python CQI table has been retired."
        )


# =============================================================================
# OpenStreetMapMCP — runtime spatial-context retrieval (external data)
# =============================================================================


class OpenStreetMapMCP(BaseMCP):
    """Spatial-context retrieval via OSM Nominatim + Overpass.

    Queries real OSM data based on station lat/lon coordinates.  This is a
    runtime data fetch, not a knowledge base, so it is exempt from the
    KG-only contract.
    """

    _NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
    _OVERPASS_URLS = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    ]
    _USER_AGENT = "TelcoAgent-Research/1.0"
    _POI_RADIUS = 500  # metres

    def __init__(self):
        super().__init__("OpenStreetMapMCP")
        self._station_coords: Dict[str, tuple] = {}
        self._cache: Dict[str, str] = {}

    def register_station(self, station_id: str, lat: float, lon: float) -> None:
        self._station_coords[station_id] = (lat, lon)

    def get_spatial_context(
        self,
        station_id: str,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> str:
        if lat is None or lon is None:
            coords = self._station_coords.get(station_id)
            if coords is None:
                return f"No coordinates registered for {station_id}."
            lat, lon = coords

        cache_key = f"{lat:.6f},{lon:.6f}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        import requests

        parts: list = []

        # 1. Nominatim reverse geocoding
        try:
            resp = requests.get(
                self._NOMINATIM_URL,
                params={
                    "lat": lat,
                    "lon": lon,
                    "format": "jsonv2",
                    "zoom": 18,
                    "addressdetails": 1,
                    "extratags": 1,
                },
                headers={"User-Agent": self._USER_AGENT},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            addr = data.get("address", {})

            loc_parts = []
            for key in ("suburb", "neighbourhood", "quarter", "city_district"):
                if key in addr:
                    loc_parts.append(addr[key])
                    break
            for key in ("city", "town", "village", "county"):
                if key in addr:
                    loc_parts.append(addr[key])
                    break
            if addr.get("state"):
                loc_parts.append(addr["state"])
            if addr.get("country"):
                loc_parts.append(addr["country"])

            area_type = data.get("type", "unknown")
            category = data.get("category", "")
            display = data.get("display_name", "")

            parts.append(f"Location: {', '.join(loc_parts) if loc_parts else display}")
            parts.append(f"Area type: {category}/{area_type}")

            road = addr.get("road", "")
            if road:
                parts.append(f"Nearest road: {road}")

        except Exception as exc:
            logger.warning("Nominatim reverse geocode failed: %s", exc)
            parts.append(f"Coordinates: ({lat:.4f}, {lon:.4f})")

        # 2. Overpass POI query (500m radius)
        try:
            query = (
                f"[out:json][timeout:30];"
                f"("
                f'  node["amenity"](around:{self._POI_RADIUS},{lat},{lon});'
                f'  node["shop"](around:{self._POI_RADIUS},{lat},{lon});'
                f'  node["landuse"](around:{self._POI_RADIUS},{lat},{lon});'
                f'  way["building"](around:{self._POI_RADIUS},{lat},{lon});'
                f'  way["landuse"](around:{self._POI_RADIUS},{lat},{lon});'
                f'  node["public_transport"](around:{self._POI_RADIUS},{lat},{lon});'
                f'  node["railway"~"station|subway_entrance|tram_stop"]'
                f"(around:{self._POI_RADIUS},{lat},{lon});"
                f'  node["highway"="bus_stop"](around:{self._POI_RADIUS},{lat},{lon});'
                f'  node["man_made"~"mast|tower"](around:{self._POI_RADIUS},{lat},{lon});'
                f'  node["tower:type"~"communication|lighting"]'
                f"(around:{self._POI_RADIUS},{lat},{lon});"
                f'  node["communication:mobile_phone"](around:{self._POI_RADIUS},{lat},{lon});'
                f");"
                f"out tags;"
            )
            resp = None
            for url in self._OVERPASS_URLS:
                try:
                    resp = requests.post(url, data={"data": query}, timeout=35)
                    resp.raise_for_status()
                    break
                except Exception:
                    logger.debug("Overpass mirror %s failed, trying next", url)
                    resp = None
            if resp is None:
                raise RuntimeError("All Overpass mirrors failed")
            elements = resp.json().get("elements", [])

            poi_counts: Dict[str, int] = {}
            landuse_types: set = set()
            building_types: set = set()
            building_heights: list = []
            transit_count: int = 0
            telecom_count: int = 0

            for el in elements:
                tags = el.get("tags", {})
                if "amenity" in tags:
                    poi_counts[tags["amenity"]] = poi_counts.get(tags["amenity"], 0) + 1
                if "shop" in tags:
                    poi_counts[f"shop:{tags['shop']}"] = (
                        poi_counts.get(f"shop:{tags['shop']}", 0) + 1
                    )
                if "landuse" in tags:
                    landuse_types.add(tags["landuse"])
                if "building" in tags:
                    building_types.add(tags.get("building", "yes"))
                    h = tags.get("height")
                    if h:
                        try:
                            building_heights.append(float(h.split()[0]))
                        except (ValueError, IndexError):
                            pass
                    elif tags.get("building:levels"):
                        try:
                            building_heights.append(float(tags["building:levels"]) * 3.0)
                        except ValueError:
                            pass
                if (
                    tags.get("public_transport")
                    or tags.get("highway") == "bus_stop"
                    or tags.get("railway") in ("station", "subway_entrance", "tram_stop")
                ):
                    transit_count += 1
                if (
                    tags.get("man_made") in ("mast", "tower")
                    or tags.get("tower:type") in ("communication", "lighting")
                    or tags.get("communication:mobile_phone")
                ):
                    telecom_count += 1

            total_pois = sum(poi_counts.values())
            top_pois = sorted(poi_counts.items(), key=lambda x: x[1], reverse=True)[:8]

            if total_pois > 0:
                density = "High" if total_pois > 30 else "Medium" if total_pois > 10 else "Low"
                parts.append(
                    f"POI density: {density} ({total_pois} POIs within {self._POI_RADIUS}m)"
                )
                poi_summary = ", ".join(f"{name}({cnt})" for name, cnt in top_pois)
                parts.append(f"Top POIs: {poi_summary}")
            else:
                parts.append(f"POI density: Very Low (no POIs within {self._POI_RADIUS}m)")

            if landuse_types:
                parts.append(f"Land use: {', '.join(sorted(landuse_types))}")

            if building_heights:
                avg_h = sum(building_heights) / len(building_heights)
                max_h = max(building_heights)
                parts.append(
                    f"Buildings: {len(building_heights)} within {self._POI_RADIUS}m, "
                    f"avg height={avg_h:.1f}m, max height={max_h:.1f}m"
                )
            elif building_types:
                top_buildings = sorted(building_types)[:5]
                parts.append(f"Buildings: types detected={', '.join(top_buildings)}")

            if transit_count > 0:
                parts.append(f"Transit hubs: {transit_count}")
            if telecom_count > 0:
                parts.append(f"Telecom infrastructure: {telecom_count} masts/towers detected")

            env_type = self._classify_environment(
                total_pois, building_heights, landuse_types, transit_count
            )
            parts.append(f"Environment hint (heuristic): {env_type}")

        except Exception as exc:
            logger.warning("Overpass POI query failed: %s", exc)

        result = "\n".join(parts)
        self._cache[cache_key] = result
        return result

    @staticmethod
    def _classify_environment(
        poi_count: int,
        building_heights: list,
        landuse_types: set,
        transit_count: int,
    ) -> str:
        avg_h = (sum(building_heights) / len(building_heights)) if building_heights else 0.0
        rural_land = landuse_types & {"farmland", "forest", "meadow", "grass", "orchard"}
        indoor_land = landuse_types & {"retail", "commercial", "industrial"}

        if rural_land and poi_count < 5 and avg_h < 10:
            return "rural / low building density"
        if avg_h > 20 and poi_count > 15:
            return "dense urban with high buildings"
        if poi_count > 10 or transit_count > 2:
            return "moderate urban street-level deployment"
        if indoor_land and poi_count > 5:
            return "indoor commercial/retail environment"
        if poi_count < 5 and avg_h < 15:
            return "sparse / low building density"
        return "default street-level deployment"
