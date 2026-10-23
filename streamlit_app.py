import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()
llm = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL_NAME = "openai/gpt-oss-120b"

st.set_page_config(page_title="Chat with your PDF", page_icon="📄")
st.title("📄 Chat with your PDF")


@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def get_client():
    return chromadb.Client()


embedder = load_embedder()
client = get_client()


def load_pdf(file):
    reader = PdfReader(file)
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


def chunk_pages(pages, size=1000, overlap=200):
    chunks, metas = [], []
    for page_num, text in pages:
        text = text.strip()
        start = 0
        while start < len(text):
            piece = text[start:start + size].strip()
            if len(piece) >= 100:
                chunks.append(piece)
                metas.append({"page": page_num})
            if start + size >= len(text):
                break
            start += size - overlap
    return chunks, metas


def build_index(chunks, metas):
    try:
        client.delete_collection("pdf_chunks")
    except Exception:
        pass
    collection = client.create_collection("pdf_chunks")
    collection.add(
        documents=chunks,
        embeddings=embedder.encode(chunks).tolist(),
        metadatas=metas,
        ids=[f"chunk_{i}" for i in range(len(chunks))],
    )


def retrieve(question, k=6):
    collection = client.get_collection("pdf_chunks")
    q_emb = embedder.encode([question]).tolist()
    result = collection.query(query_embeddings=q_emb, n_results=k)
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    return [(d, m["page"]) for d, m in zip(docs, metas)]


def build_prompt(question, results):
    context = "\n\n---\n\n".join(f"[Page {p}]\n{d}" for d, p in results)
    return f"""You are answering questions about a document.
Use ONLY the excerpts below. Excerpts may be cut off mid-sentence and may come from different parts of the document, so combine them.
Many excerpts may come from a literature review, where the authors describe weaknesses of OTHER, earlier methods. Do NOT report those as limitations of the authors' own proposed system. Only report limitations the authors state about their own work, usually in the conclusion or future-work sections. If an excerpt's source is unclear, say so.
After each key point, cite the page like (p. 12).
If the excerpts really do not contain the answer, say "I couldn't find this in the document."

Excerpts:
{context}

Question: {question}"""


def ask(question):
    results = retrieve(question)
    response = llm.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[{"role": "user", "content": build_prompt(question, results)}],
    )
    return response.choices[0].message.content, results


def show_sources(results):
    with st.expander("Sources"):
        for text, page in results:
            st.markdown(f"**Page {page}**")
            st.write(text)


if "messages" not in st.session_state:
    st.session_state.messages = []
if "file_name" not in st.session_state:
    st.session_state.file_name = None

uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded is None:
    st.info("Upload a PDF to start chatting.")
    st.stop()

if uploaded.name != st.session_state.file_name:
    pages = load_pdf(uploaded)
    if not any(t.strip() for _, t in pages):
        st.error("No text found. This PDF may be scanned (image-only).")
        st.stop()
    chunks, metas = chunk_pages(pages)
    build_index(chunks, metas)
    st.session_state.file_name = uploaded.name
    st.session_state.messages = []
    empty_pages = sum(1 for _, t in pages if not t.strip())
    st.success(f"Indexed {len(chunks)} chunks from {len(pages)} pages of {uploaded.name}")
    if empty_pages:
        st.warning(f"{empty_pages} page(s) had no extractable text (blank or image-only) and were skipped.")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            show_sources(m["sources"])

question = st.chat_input("Ask something about your PDF")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = ask(question)
        st.markdown(answer)
        show_sources(sources)
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )