import os
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer

def load_pdf(path):
    reader = PdfReader(path)
    return " ".join(page.extract_text() for page in reader.pages)

def chunk_text(text, size=500, overlap=100):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks

text = load_pdf("sample.pdf")
chunks = chunk_text(text)

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client()
collection = client.create_collection("pdf_chunks")
collection.add(
    documents=chunks,
    embeddings=model.encode(chunks).tolist(),
    ids=[f"chunk_{i}" for i in range(len(chunks))],
)

def retrieve(question, k=2):
    q_emb = model.encode([question]).tolist()
    result = collection.query(query_embeddings=q_emb, n_results=k)
    return result["documents"][0]

questions = [
    "Where did she study?",
    "What is her research experience?",
]

for q in questions:
    print("\n==========================")
    print("QUESTION:", q)
    for i, chunk in enumerate(retrieve(q), 1):
        print(f"--- Result {i} ---")
        print(chunk)

load_dotenv()
llm = Groq(api_key=os.getenv("GROQ_API_KEY"))

def build_prompt(question, context_chunks):
    context = "\n\n".join(context_chunks)
    return f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I couldn't find this in the document."

Context:
{context}

Question: {question}"""

def ask(question):
    chunks = retrieve(question, k=4)
    response = llm.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": build_prompt(question, chunks)}],
    )
    return response.choices[0].message.content

for q in ["Where did she study?", "What programming languages does she know?"]:
    print("\nQUESTION:", q)
    print("ANSWER:", ask(q))       