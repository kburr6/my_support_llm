# download_model.py
from sentence_transformers import SentenceTransformer
import os

# The full name is needed for downloading
model_name = "BAAI/bge-large-en-v1.5"
# The simple, local path where we will save it
save_path = "/app/embedding_model"

print(f"Dockerfile: Downloading embedding model '{model_name}' to {save_path}...")
# The .save() method is more direct than using the cache.
model = SentenceTransformer(model_name)
model.save(save_path)
print("Dockerfile: Download complete.")