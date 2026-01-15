print("Before import")
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

os.makedirs("data" , exist_ok=True)
from sentence_transformers import SentenceTransformer
print("After import")

import numpy as np

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Model loaded")

with open("docs/doc1.txt", "r") as f:

    documents = [line.strip() for line in f if line.strip()]

embedding = model.encode(
    documents,
    convert_to_numpy=True,
    show_progress_bar=True
)


with open("data", "w") as f:
    np.save("data/embeddings.npy", embedding)
print("Embedding shape:", embedding.shape)
