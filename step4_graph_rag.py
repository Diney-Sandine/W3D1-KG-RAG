import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain

# 1. Load environment variables
load_dotenv()

# 2. Connect to Neo4j
print("Connecting to Neo4j...")
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
)
print("✅ Connected to Neo4j successfully!")

# 3. Initialize ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

# 4. Custom Cypher generation prompt tailored to your actual schema
CYPHER_GENERATION_TEMPLATE = """
Task:
Generate a Cypher query to query a Neo4j graph database based on the user's question.
The graph has the following node labels and properties:
- (:Scheme {{id: STRING}})
- (:Department {{id: STRING}})
- (:Concept {{id: STRING}})
- (:Document {{id: STRING, title: STRING, text: STRING, source: STRING}})
- (:Program {{id: STRING}})

Relationships:
- (:Department)-[:INCLUDES]->(:Scheme)
- (:Scheme)-[:RELATED_TO]->(:Program)

Guidelines:
- Match scheme names or concepts using case-insensitive CONTAINS on `id` properties (e.g., `WHERE s.id CONTAINS 'seed'` or `WHERE c.id CONTAINS 'training'`).
- Return relevant nodes clearly.

Schema:
{schema}

Question:
{question}

Cypher Query:
"""

from langchain_core.prompts import PromptTemplate
CYPHER_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)

# 5. Create the Graph Cypher QA Chain with the custom prompt
chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    cypher_prompt=CYPHER_PROMPT,
    verbose=True,
    allow_dangerous_requests=True
)

# 6. Test a query against your Knowledge Graph
question = "What are the agricultural schemes available for seeds or training?"
print(f"\n❓ Asking Knowledge Graph: '{question}'\n")

response = chain.invoke({"query": question})
print("🤖 Answer:")
print(response["result"])
