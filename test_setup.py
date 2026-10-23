from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer

print("Imports OK")

model = SentenceTransformer("all-MiniLM-L6-v2")
vec = model.encode(["hello world"])
print("Embedding shape:", vec.shape)