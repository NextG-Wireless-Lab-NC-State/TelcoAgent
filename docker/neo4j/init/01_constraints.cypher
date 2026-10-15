// TelcoAgent KG constraints (Neo4j 5.x syntax)
// Mounted to /var/lib/neo4j/import/init/ in the neo4j container.
// Apply with: cypher-shell -u neo4j -p $NEO4J_PASSWORD -f /var/lib/neo4j/import/init/01_constraints.cypher
// (Phase 3 will invoke this from telcoagent.stores.ontology_store._ensure_kg_indexes.)

CREATE CONSTRAINT spec_id_unique IF NOT EXISTS
  FOR (s:Spec) REQUIRE s.spec_id IS UNIQUE;

CREATE CONSTRAINT section_id_unique IF NOT EXISTS
  FOR (s:Section) REQUIRE s.section_id IS UNIQUE;

CREATE CONSTRAINT page_id_unique IF NOT EXISTS
  FOR (p:Page) REQUIRE (p.spec_id, p.page_no) IS UNIQUE;

CREATE CONSTRAINT figure_id_unique IF NOT EXISTS
  FOR (f:Figure) REQUIRE f.figure_id IS UNIQUE;

CREATE CONSTRAINT concept_id_unique IF NOT EXISTS
  FOR (c:Concept) REQUIRE c.concept_id IS UNIQUE;

CREATE INDEX page_sha256 IF NOT EXISTS
  FOR (p:Page) ON (p.sha256);

CREATE INDEX figure_caption_text IF NOT EXISTS
  FOR (f:Figure) ON (f.caption);
