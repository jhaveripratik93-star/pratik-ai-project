import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

def setup_chroma(text_data=None, persist_dir="./chroma_persist"):
    client = chromadb.Client(
        Settings(
            persist_directory=persist_dir,
            anonymized_telemetry=False
        )
    )

    ef = embedding_functions.HuggingFaceEmbeddingFunction(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    collection = client.get_or_create_collection(
        name="logs_collection",
        embedding_function=ef
    )

    if text_data:
        ids = [f"doc{i}" for i in range(len(text_data))]
        collection.add(ids=ids, documents=text_data)

    return client, collection
