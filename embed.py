from sentence_transformers import SentenceTransformer
import numpy as np




model = SentenceTransformer("sentence-transformer/all-MiniLM-L6-v2")


with open("doc.txt" , mode = 'r') as f:
    documents = [line.strip() for line in f if line.strip()]

embedding = model.encode(
    documents,
    convert_to_numpy = True,
    show_progress_bar = True
)

np.save("data/emebeddin.npy" , embedding)

print(f"Embedding Shape {embedding.shape}")