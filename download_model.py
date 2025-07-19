# download_model.py
from sentence_transformers import SentenceTransformer
# We will get the model name from an environment variable during the build
import os
model_name = os.environ.get("EMBEDDING_MODEL_NAME", "BAAI/bge-large-en-v1.5")
print(f"Dockerfile: Downloading embedding model '{model_name}'...")
SentenceTransformer(model_name, cache_folder='./models')
print("Dockerfile: Download complete.")
