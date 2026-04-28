from neo4j import GraphDatabase
from google import genai
from pydantic import BaseModel, Field

# --- Config ---

URI = "neo4j://127.0.0.1:7687"
AUTH = ("neo4j", "demodemo")

USER_QUERY = "Who should I contact for this year's HackIllinois event?"

client = genai.Client(api_key=GEMINI_API_KEY)
driver = GraphDatabase.driver(URI, auth=AUTH)

# --- Step 1: Get the graph schema so Gemini knows what's available ---
with driver:
    labels_result = driver.execute_query(
        "CALL db.labels() YIELD label RETURN collect(label) AS labels",
        database_="neo4j",
    )
    rel_result = driver.execute_query(
        "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) AS types",
        database_="neo4j",
    )
    prop_result = driver.execute_query(
        """
        MATCH (n)
        WITH labels(n)[0] AS label, keys(n) AS props
        RETURN label, collect(DISTINCT props) AS sample_props
        LIMIT 20
        """,
        database_="neo4j",
    )

    labels = labels_result.records[0]["labels"]
    rel_types = rel_result.records[0]["types"]
    schema_lines = [f"Node labels: {labels}", f"Relationship types: {rel_types}"]
    for r in prop_result.records:
        # Flatten the list of property key lists
        all_props = sorted(set(k for prop_list in r["sample_props"] for k in prop_list))
        schema_lines.append(f"  :{r['label']} properties: {all_props}")
    schema_str = "\n".join(schema_lines)

    print("=== Graph Schema ===")
    print(schema_str)
    print()

    # --- Step 2: Ask Gemini to generate a Cypher query ---

    class CypherQuery(BaseModel):
        reasoning: str = Field(description="Brief explanation of why this query answers the user's question")
        cypher: str = Field(description="A valid Cypher READ query (no writes). Use RETURN to return relevant nodes/relationships.")

    print(f"User query: {USER_QUERY}\n")
    print("Generating Cypher query with Gemini...")

    cypher_response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=f"""You are a Neo4j Cypher expert. Given a user's question and a graph schema,
generate a Cypher query that retrieves the relevant information.

Graph schema:
{schema_str}

User question: {USER_QUERY}

Rules:
- Only generate READ queries (MATCH/RETURN), never write/delete
- Return enough context to answer the question (node names, properties, relationships)
- Keep the query focused but include related nodes that provide useful context
- If the question is about people/contacts, include their roles, organizations, and contact info if available""",
        config={
            "response_mime_type": "application/json",
            "response_json_schema": CypherQuery.model_json_schema(),
        },
    )

    cypher_result = CypherQuery.model_validate_json(cypher_response.text)
    print(f"Reasoning: {cypher_result.reasoning}")
    print(f"Cypher: {cypher_result.cypher}\n")

    # --- Step 3: Execute the Cypher query against Neo4j ---
    print("Querying Neo4j...")
    try:
        query_result = driver.execute_query(cypher_result.cypher, database_="neo4j")
        context_rows = []
        for record in query_result.records:
            row = {key: str(record[key]) for key in record.keys()}
            context_rows.append(row)
        context_str = "\n".join(str(row) for row in context_rows)
        print(f"Got {len(context_rows)} result(s) from the graph.\n")
    except Exception as e:
        print(f"Cypher query failed: {e}")
        print("Falling back to broad context query...\n")
        query_result = driver.execute_query(
            """
            MATCH (n)
            OPTIONAL MATCH (n)-[r]->(m)
            RETURN labels(n)[0] AS type, n.name AS name, properties(n) AS props,
                   type(r) AS rel, m.name AS related_to
            LIMIT 50
            """,
            database_="neo4j",
        )
        context_rows = []
        for record in query_result.records:
            row = {key: str(record[key]) for key in record.keys()}
            context_rows.append(row)
        context_str = "\n".join(str(row) for row in context_rows)
        print(f"Got {len(context_rows)} result(s) from fallback query.\n")

    # --- Step 4: Use graph results as context for the final answer ---
    print("Generating answer with graph context...\n")
    print("=" * 60)

    answer_response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=f"""You are a helpful assistant. Use ONLY the following knowledge graph context
to answer the user's question. If the context doesn't contain enough information,
say so honestly.

Knowledge graph context (from Neo4j):
{context_str}

User question: {USER_QUERY}

Provide a clear, helpful answer based on the graph data above.""",
    )

    print(f"Q: {USER_QUERY}\n")
    print(f"A: {answer_response.text}")
    print("=" * 60)
