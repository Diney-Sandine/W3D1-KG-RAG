import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load environment variables from .env
load_dotenv()

# 2. Initialize & test the Neo4j Graph connection
print("Connecting to Neo4j...")
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
)
print("✅ Connected to Neo4j successfully!")

# 3. Load the Tamil Nadu Agriculture Schemes webpage
url = "https://www.tn.gov.in/scheme_list.php?dep_id=Mg=="
print(f"\nLoading webpage: {url}")

loader = WebBaseLoader(
    web_paths=[url],
    requests_kwargs={
        "headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        },
        "verify": False  # Avoids occasional SSL verification issues on .gov.in sites
    }
)

raw_docs = loader.load()
print(f"✅ Loaded {len(raw_docs)} document(s).")

# 4. Split the webpage content into manageable chunks for Entity Extraction
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(raw_docs)

print(f"✅ Split webpage into {len(chunks)} chunks.")
print("\n--- Sample Chunk 1 Preview ---")
print(chunks[0].page_content[:500])
