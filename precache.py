# precache.py
from langchain_huggingface import HuggingFaceEmbeddings

print("Starting model pre-caching...")
try:
    # This line will download and cache the model
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("Model has been successfully pre-cached.")
except Exception as e:
    print(f"An error occurred during pre-caching: {e}")