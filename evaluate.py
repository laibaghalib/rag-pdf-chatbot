from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer

PDF_PATH = "paper.pdf"
K = 6

# Har sawal ke saath wo lafz jo sahi jawab wale chunk mein hona chahiye
TEST_SET = [
    {"question": "What dataset was used?", "answer": "Kaggle"},
    {"question": "What image resolution was used for training?", "answer": "128 x 128"},
    {"question": "What optimizer was used?", "answer": "Adam"},
    {"question": "How many epochs was the model trained for?", "answer": "100 epochs"},
    {"question": "What software framework was used?", "answer": "TensorFlow"},
    {"question": "What was the mean PSNR?", "answer": "32.7"},
    {"question": "What was the SSIM value?", "answer": "0.95"},
    {"question": "What dropout rate was used?", "answer": "0.3"},
    {"question": "What are the steps of the proposed pipeline?", "answer": "five-step"},
    {"question": "What data augmentation was used?", "answer": "flip"},
    {"question": "What are the future work suggestions?", "answer": "512"},
    {"question": "Why is gadolinium contrast a risk for patients?", "answer": "accumulation"},
]


def norm(s):
    return "".join(s.lower().split())


model = SentenceTransformer("all-MiniLM-L6-v2")
reader = PdfReader(PDF_PATH)
pages = [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]


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


chunks, metas = chunk_pages(pages)
client = chromadb.Client()
collection = client.create_collection("eval")
collection.add(
    documents=chunks,
    embeddings=model.encode(chunks).tolist(),
    metadatas=metas,
    ids=[f"c{i}" for i in range(len(chunks))],
)

hits = 0
for item in TEST_SET:
    q_emb = model.encode([item["question"]]).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=K)
    docs = res["documents"][0]
    found_pages = [
        m["page"] for d, m in zip(docs, res["metadatas"][0])
        if norm(item["answer"]) in norm(d)
    ]
    ok = len(found_pages) > 0
    hits += ok
    print(("PASS" if ok else "FAIL"), "|", item["question"])
    print("   looking for:", item["answer"], "| found on pages:", found_pages)

print(f"\nAnswer-in-context rate @k={K}: {hits}/{len(TEST_SET)} = {hits / len(TEST_SET):.0%}")