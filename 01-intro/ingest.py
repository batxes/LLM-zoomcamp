from elasticsearch import Elasticsearch
from tqdm.auto import tqdm

# Health check
es_client = Elasticsearch("http://127.0.0.1:9200")
es_client.info()

# Create index
index_settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"},
            "course": {"type": "keyword"} 
        }
    }
}
index_name = "course_questions"
es_client.indices.create(index=index_name, body=index_settings)

# Ingest documents
documents = get_documents()
for doc in tqdm(documents):
    es_client.index(index=index_name, document=doc)
