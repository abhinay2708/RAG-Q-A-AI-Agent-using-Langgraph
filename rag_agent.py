from typing import TypedDict
from pathlib import Path
import os

# LangChain + vector DB + embeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

# LangGraph
from langgraph.graph import StateGraph, END

# Gemini SDK
import google.generativeai as genai

import warnings
warnings.filterwarnings("ignore")

# enable tracing

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "AI Agent using LangGraph" 

LANGSMITH_API_KEY=os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY

# STATE DEFINITION

class AgentState(TypedDict):
    question: str
    context: str
    answer: str
    reflection: str
    
# DOCUMENT LOADING

def load_documents(folder="data"):
    folder = Path(folder)

    docs = []
    for file in folder.glob("*.txt"):
        text = file.read_text(encoding="utf-8")
        print(f"[LOAD] Loaded {file.name}")
        docs.append(text)
    return docs

# BUILD VECTOR DATABASE

def build_vectordb(texts):
    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_text("\n\n".join(texts))

    print(f"[VDB] Created {len(chunks)} text chunks")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_texts(chunks, embedding=embeddings)
    return vectordb

# LOAD GEMINI
def load_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Please set GEMINI_API_KEY environment variable.")

    genai.configure(api_key=api_key)
    print("Using Gemini 2.0 Model")
    return genai.GenerativeModel("gemini-2.0-flash")
GEMINI = load_gemini()

# Plan Node
def plan_node(state: AgentState):
    q = state["question"]
    qwords = ["what", "why", "how", "benefit", "explain", "Tell me about"]

    need = any(k in q.lower() for k in qwords)
    print("\n[PLAN] retrieval needed?", need)

    return {"context": "RETRIEVE" if need else ""}

# Retrieve Node
def retrieve_node(state: AgentState):
    print("[RETRIEVE] retrieving chunks from vector DB...")
    results = vectordb.similarity_search(state["question"], k=1)

    context = "\n\n---\n\n".join([r.page_content for r in results]) if results else ""
    return {"context": context}

# Answer Node
def answer_node(state: AgentState):
    q = state["question"]
    c = state["context"]

    prompt = f"""
Use ONLY the context to answer the question.

Context:
{c}

Question:
{q}

Answer:
"""

    response = GEMINI.generate_content(prompt)
    answer = response.text

    print("[ANSWER]", answer[:80], "...")
    return {"answer": answer}

# Reflect Node
def reflect_node(state: AgentState):
    q = state["question"]
    a = state["answer"]

    prompt = f"""
Rate the relevance of the answer from 0 to 10. Then explain briefly.

Question: {q}
Answer: {a}

Format:
Score: <0-10>
Reason: <one sentence>
"""

    response = GEMINI.generate_content(prompt)
    review = response.text

    print("[REFLECT]", review[:80].replace("\n", " "), "...")
    return {"reflection": review}

# BUILD LANGGRAPH
graph = StateGraph(AgentState)

graph.add_node("plan", plan_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("answer", answer_node)
graph.add_node("reflect", reflect_node)

graph.set_entry_point("plan")

graph.add_conditional_edges(
    "plan",
    lambda s: "retrieve" if s["context"] == "RETRIEVE" else "answer",
    {"retrieve": "retrieve", "answer": "answer"},
)

graph.add_edge("retrieve", "answer")
graph.add_edge("answer", "reflect")
graph.add_edge("reflect", END)

app = graph.compile()

docs = load_documents()
vectordb = build_vectordb(docs)


def ask(question: str):
    """
    Runs the LangGraph RAG workflow and returns:
    {question, context, answer, reflection}
    """
    input_state = {
        "question": question,
        "context": "",
        "answer": "",
        "reflection": "",
    }

    result = app.invoke(input_state)
    return result