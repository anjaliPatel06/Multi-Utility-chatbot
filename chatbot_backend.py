from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode, tools_condition
import requests

load_dotenv()

# -------------------
# 1. LLM + embeddings
# -------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------
# 2. PDF retriever store (per thread)
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


# -------------------
# 3. Tools — RAG is NOT in this list (handled directly in chat_node)
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given ticker symbol (e.g. 'AAPL', 'TSLA').
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}"
        f"&apikey={os.getenv('ALPHA_VANTAGE_API_KEY')}"
    )
    r = requests.get(url)
    return r.json()


# Only these tools are exposed to the LLM
llm_tools = [search_tool, calculator, get_stock_price]
llm_with_tools = llm.bind_tools(llm_tools)


# -------------------
# 4. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# 5. RAG helper — called directly by chat_node, never by LLM
# -------------------
def _run_rag(query: str, thread_id: str) -> str:
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return "No document indexed for this chat. Please upload a PDF first."

    docs = retriever.invoke(query)
    if not docs:
        return "No relevant content found in the document."

    context_parts = [f"[Chunk {i}]\n{doc.page_content}" for i, doc in enumerate(docs, 1)]
    source = _THREAD_METADATA.get(thread_id, {}).get("filename", "uploaded PDF")
    return f"Relevant content from '{source}':\n\n" + "\n\n".join(context_parts)


# Keywords that mean the user wants a TOOL, NOT the document
_TOOL_OVERRIDE_KEYWORDS = [
    # stock
    "stock price", "share price", "stock value", "market price",
    "ticker", "nasdaq", "nyse", "trading at", "current price of",
    # math
    "calculate", "multiply", "divide", "add", "subtract",
    "plus", "minus", "times", "divided by", "percent of",
    "square root", "how much is", "what is the sum",
    # web search
    "latest news", "current news", "today", "weather", "score",
    "who won", "search the web", "look up online",
]

# Keywords that strongly indicate a document question
_DOC_KEYWORDS = [
    "document", "pdf", "uploaded", "file", "the text",
    "according to the", "from the doc", "in the doc",
    "page ", "pages", "summarize", "summary of",
    "what does the doc", "what does the file",
    "mentioned in", "written in", "stated in",
    "based on the document", "based on the pdf",
    "from the uploaded",
]


def _is_rag_query(user_message: str, has_document: bool) -> bool:
    """
    Return True ONLY if:
      - A document is loaded, AND
      - The message contains a document-specific keyword, AND
      - The message does NOT look like a stock / math / web-search question.
    """
    if not has_document:
        return False

    lower = user_message.lower()

    # If the message clearly wants a tool → never go to RAG
    if any(kw in lower for kw in _TOOL_OVERRIDE_KEYWORDS):
        return False

    # Otherwise check for explicit document keywords
    return any(kw in lower for kw in _DOC_KEYWORDS)


# -------------------
# 6. Nodes
# -------------------
def chat_node(state: ChatState, config=None):
    """
    Main LLM node.
    - Explicit document questions  → RAG path (direct Python call, no LLM tool)
    - Everything else              → LLM decides (tool or direct answer)
    """
    thread_id = ""
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id", "")

    has_document = thread_id in _THREAD_RETRIEVERS

    # Find the latest human message
    last_human_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_human_msg = msg.content if isinstance(msg.content, str) else ""
            break

    # ── RAG path ────────────────────────────────────────────────────────────
    if last_human_msg and _is_rag_query(last_human_msg, has_document):
        rag_context = _run_rag(last_human_msg, thread_id)

        rag_system = SystemMessage(
            content=(
                "You are a helpful assistant. Answer the user's question using ONLY "
                "the document context provided below. Be clear and concise.\n\n"
                f"{rag_context}"
            )
        )
        response = llm.invoke([rag_system, *state["messages"]])
        return {"messages": [response]}

    # ── Normal path (tools / direct answer) ─────────────────────────────────
    doc_status = (
        f"A PDF named '{_THREAD_METADATA[thread_id]['filename']}' is indexed for this chat. "
        "If the user explicitly asks about the document/PDF content, let them know to "
        "phrase it with words like 'from the document' or 'in the PDF'."
        if has_document
        else "No PDF uploaded for this chat."
    )

    system_message = SystemMessage(
        content=f"""You are a helpful AI assistant.

DOCUMENT STATUS: {doc_status}

Available tools:
- calculator        : arithmetic (add, sub, mul, div) — use for any math question
- get_stock_price   : real-time stock price by ticker symbol (e.g. AAPL, TSLA, GOOGL)
- duckduckgo_search : web search for current events, news, general knowledge

STRICT RULES:
- Stock price question (any company/ticker)  → ALWAYS use get_stock_price tool
- Math / arithmetic question                 → ALWAYS use calculator tool
- Current events / web info                 → use duckduckgo_search tool
- Simple factual / conversational           → answer directly, no tool needed
- Document / PDF content questions          → handled separately, just tell user to ask with "from the document"
- Use ONLY ONE tool per turn
- Never fabricate tool arguments
"""
    )

    response = llm_with_tools.invoke([system_message, *state["messages"]], config=config)
    return {"messages": [response]}


tool_node = ToolNode(llm_tools)

# -------------------
# 7. Checkpointer
# -------------------
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# -------------------
# 8. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)


# -------------------
# 9. Helpers
# -------------------
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})