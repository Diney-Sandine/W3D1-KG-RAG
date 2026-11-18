import os
import asyncio
from typing import AsyncGenerator
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph
from langchain_core.messages import HumanMessage, AIMessage

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

# 3. Initialize ChatOpenAI with streaming enabled
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

# Global in-memory chat history per session ID
chat_histories = {}


def contextualize_question(question: str, history: list) -> str:
    """Uses chat history to turn follow-up questions into standalone questions."""
    if not history:
        return question
    
    history_text = "\n".join(
        [f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}" for m in history[-6:]]
    )
    rewrite_prompt = (
        f"Given the conversation history below, rewrite the follow-up question into a standalone question "
        f"that includes all necessary context. Do not answer it, only return the rewritten question.\n\n"
        f"Chat History:\n{history_text}\n\nFollow-up Question: {question}\nStandalone Question:"
    )
    return llm.invoke(rewrite_prompt).content.strip()


async def ask_kg_rag_stream(question: str, session_id: str = "default_farmer") -> AsyncGenerator[str, None]:
    """Queries Neo4j Knowledge Graph manually and streams response tokens with memory."""
    if session_id not in chat_histories:
        chat_histories[session_id] = []

    history = chat_histories[session_id]
    full_response = ""

    try:
        # Step A: Contextualize follow-up question using memory
        standalone_q = contextualize_question(question, history)

        # Step B: Generate Cypher query using LLM based on schema
        cypher_prompt = f"""
        Given the following Neo4j schema:
        {graph.schema}

        Write a valid Neo4j Cypher query to answer this user question: "{standalone_q}"
        Return ONLY the Cypher query string inside standard text (no markdown code blocks if possible, or just the query).
        """
        cypher_query = llm.invoke(cypher_prompt).content.replace("```cypher", "").replace("```", "").strip()

        # Step C: Execute Cypher query against Neo4j
        graph_data = []
        try:
            graph_data = graph.query(cypher_query)
        except Exception as cypher_err:
            graph_data = f"Could not execute query directly: {str(cypher_err)}"

        # Step D: Synthesize final answer using LLM with graph context
        answer_prompt = f"""
        You are **Uzhavan Sahayak (உழவன் சகாயக்)**, an expert AI assistant for Tamil Nadu farmers.
        Answer the user's question using the retrieved Knowledge Graph data below.
        If the data is empty or irrelevant, state clearly what schemes are available or explain based on general knowledge of the TN agriculture department.

        User Question: {standalone_q}
        Retrieved Graph Data: {graph_data}
        """
        
        response_stream = llm.stream(answer_prompt)

        # Step E: Stream the response token-by-token
        for chunk in response_stream:
            token = chunk.content
            if token:
                full_response += token
                yield token
                await asyncio.sleep(0.02)

        # Step F: Save conversation turn to memory
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=full_response.strip()))

    except Exception as e:
        yield f"An error occurred while querying the Knowledge Graph: {str(e)}"


# 4. Interactive CLI Chat Loop to test Memory & Streaming
async def main():
    print("\n🌾 Tamil Nadu Agriculture KG-RAG Chatbot Ready! (Type 'exit' to quit)\n")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            break
        if not user_input.strip():
            continue

        print("Bot: ", end="", flush=True)
        async for token in ask_kg_rag_stream(user_input, session_id="farmer_1"):
            print(token, end="", flush=True)
        print("\n")

if __name__ == "__main__":
    asyncio.run(main())
    