import json
import nest_asyncio
nest_asyncio.apply()

from llama_parse import LlamaParse
from neo4j import GraphDatabase
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List

# --- Config ---

URI = "neo4j://127.0.0.1:7687"
AUTH = ("neo4j", "demodemo")
PDF_PATH = "open-ceremony.pdf"

# --- Step 1: Parse the PDF with LlamaParse ---
print("Parsing PDF with LlamaParse...")
parser = LlamaParse(
    api_key=LLAMAPARSE_KEY,
    result_type="markdown",
    verbose=True,
)
documents = parser.load_data(PDF_PATH)
full_text = "\n".join(doc.text for doc in documents)
print(f"Parsed {len(documents)} document(s), {len(full_text)} chars total.\n")

# --- Step 2: Use Gemini to extract nodes and edges ---

class Node(BaseModel):
    id: str = Field(description="Unique identifier for the node (lowercase_snake_case)")
    label: str = Field(description="The node label/type (e.g. Person, Event, Location, Organization)")
    name: str = Field(description="Human-readable name of the entity")
    properties: dict = Field(default_factory=dict, description="Additional properties for the node")

class Edge(BaseModel):
    source: str = Field(description="The id of the source node")
    target: str = Field(description="The id of the target node")
    relationship: str = Field(description="The relationship type in UPPER_SNAKE_CASE (e.g. PERFORMED_AT, ORGANIZED_BY)")
    properties: dict = Field(default_factory=dict, description="Additional properties for the edge")

class KnowledgeGraph(BaseModel):
    nodes: List[Node] = Field(description="All entities extracted as graph nodes")
    edges: List[Edge] = Field(description="All relationships between nodes")

print("Sending parsed text to Gemini for graph extraction...")
client = genai.Client(api_key=GEMINI_API_KEY)

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=f"""Analyze the following document and extract a knowledge graph from it.

Identify all important entities (people, places, events, organizations, concepts, dates, etc.)
and the relationships between them.

Rules:
- Each node must have a unique id in lowercase_snake_case
- Each node must have a label (its type, e.g. Person, Event, Location, Organization, Concept)
- Each edge must reference existing node ids as source and target
- Relationship types should be in UPPER_SNAKE_CASE (e.g. PARTICIPATED_IN, LOCATED_AT, ORGANIZED_BY)
- Extract as many meaningful entities and relationships as you can find
- Include relevant properties on nodes (e.g. dates, descriptions, roles)

Document:
{full_text}""",
    config={
        "response_mime_type": "application/json",
        "response_json_schema": KnowledgeGraph.model_json_schema(),
    },
)

graph = KnowledgeGraph.model_validate_json(response.text)
print(f"Extracted {len(graph.nodes)} nodes and {len(graph.edges)} edges.\n")

# --- Step 3: Insert into Neo4j ---
driver = GraphDatabase.driver(URI, auth=AUTH)

with driver:
    # Clear previous run
    driver.execute_query("MATCH (n) DETACH DELETE n", database_="neo4j")

    # Create nodes
    for node in graph.nodes:
        props = {**node.properties, "name": node.name, "id": node.id}
        # Cypher doesn't allow parameterized labels, so we sanitize and inject it
        label = "".join(c for c in node.label if c.isalnum() or c == "_")
        prop_keys = ", ".join(f"{k}: ${k}" for k in props)
        driver.execute_query(
            f"CREATE (n:{label} {{{prop_keys}}})",
            **props,
            database_="neo4j",
        )

    # Create edges
    for edge in graph.edges:
        rel_type = "".join(c for c in edge.relationship if c.isalnum() or c == "_")
        prop_str = ""
        params = {"source_id": edge.source, "target_id": edge.target, **edge.properties}
        if edge.properties:
            prop_keys = ", ".join(f"{k}: ${k}" for k in edge.properties)
            prop_str = f" {{{prop_keys}}}"
        driver.execute_query(
            f"""
            MATCH (a {{id: $source_id}}), (b {{id: $target_id}})
            CREATE (a)-[:{rel_type}{prop_str}]->(b)
            """,
            **params,
            database_="neo4j",
        )

    # Print summary
    node_result = driver.execute_query(
        "MATCH (n) RETURN labels(n)[0] AS label, n.name AS name, n.id AS id ORDER BY label, name",
        database_="neo4j",
    )
    edge_result = driver.execute_query(
        "MATCH (a)-[r]->(b) RETURN a.name AS from, type(r) AS rel, b.name AS to",
        database_="neo4j",
    )

    print("=== Nodes ===")
    for r in node_result.records:
        print(f"  [{r['label']}] {r['name']} ({r['id']})")

    print(f"\n=== Edges ===")
    for r in edge_result.records:
        print(f"  {r['from']} --{r['rel']}--> {r['to']}")

    print(f"\nGraph created: {len(node_result.records)} nodes, {len(edge_result.records)} edges.")
