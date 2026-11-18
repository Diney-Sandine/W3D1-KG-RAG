import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader

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

# 3. Load the Tamil Nadu Agriculture Schemes webpage data again
url = "https://www.tn.gov.in/scheme_list.php?dep_id=Mg=="
print(f"Loading data from {url}...")

loader = WebBaseLoader(
    web_paths=[url],
    requests_kwargs={
        "headers": {"User-Agent": "Mozilla/5.0"},
        "verify": False
    }
)
raw_docs = loader.load()

# 4. Split text into chunks so the LLM can process them effectively
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
chunks = text_splitter.split_documents(raw_docs)
# Limit to first few chunks for a quick initial run if desired (e.g., chunks[:5])
docs_to_process = chunks[:5] 
print(f"✅ Split data into chunks. Processing first {len(docs_to_process)} chunks into Knowledge Graph...")

# 5. Initialize ChatOpenAI and LLMGraphTransformer
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_transformer = LLMGraphTransformer(llm=llm)

# 6. Convert text documents into Graph Documents (Entities & Relationships)
print("Extracting entities and relationships using OpenAI...")
graph_documents = llm_transformer.convert_to_graph_documents(docs_to_process)
print(f"✅ Extracted {len(graph_documents)} graph document structure(s).")

# 7. Store the graph documents into Neo4j
print("Saving graph documents to Neo4j database...")
graph.add_graph_documents(
    graph_documents,
    baseEntityLabel=True,
    include_source=True
)

print("🎉 Success! Knowledge Graph populated in Neo4j.")
