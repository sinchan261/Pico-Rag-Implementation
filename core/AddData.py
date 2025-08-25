# core/AddData.py
import os
import json
from core.rag_engine import RAGengine

rag = RAGengine()

def AddData():
    # ✅ Always resolve absolute path for RagData folder
    base_dir = os.path.dirname(os.path.abspath(__file__))  # -> SmartAiFriend_Pico/core
    folder_path = os.path.join(base_dir, "..", "RagData")  # -> SmartAiFriend_Pico/RagData
    folder_path = os.path.abspath(folder_path)

    if not os.path.exists(folder_path):
        print(f"❌ RagData folder not found at: {folder_path}")
        return

    total_docs = 0
    # ✅ Loop through all JSON files
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    documents = json.load(f)

                for doc in documents:
                    rag.add_document(doc["id"], doc["text"])

                print(f"✅ Loaded {len(documents)} docs from {filename}")
                total_docs += len(documents)

            except Exception as e:
                print(f"⚠️ Error loading {filename}: {e}")

    print(f"🎉 All JSON files processed successfully! Total docs: {total_docs}")
