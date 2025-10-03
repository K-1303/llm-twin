from qdrant_client import QdrantClient
import json
from llm_engineering.settings import settings

# Connect to Qdrant
client = QdrantClient(
            url=settings.QDRANT_CLOUD_URL,
            api_key=settings.QDRANT_APIKEY,
        )

# Get all points from a collection
collection_name1 = "cleaned_articles"
collection_name2 = "cleaned_posts"
collection_naame3 = "cleaned_repositories"

# Scroll through all points (recommended for large datasets)
def extract_all_documents(client, collection_name):
    all_points = []
    offset = None
    
    while True:
        # Scroll through points in batches
        result = client.scroll(
            collection_name=collection_name,
            limit=100,  # Batch size
            offset=offset,
            with_payload=True,
            with_vectors=True  # Set to False if you don't need vectors
        )
        
        points, next_offset = result
        all_points.extend(points)
        
        if next_offset is None:
            break
        offset = next_offset
    
    return all_points

# Extract all documents
all_documents = extract_all_documents(client, collection_name1)
all_documents += extract_all_documents(client, collection_name2)
all_documents += extract_all_documents(client, collection_naame3)

# Convert to JSON
json_output = []
for point in all_documents:
    doc = {
        "id": point.id,
        "payload": point.payload,
        "vector": point.vector  # Remove this line if you don't need vectors
    }
    json_output.append(doc)

# Save to file
with open("cleaned_documents.json", "w", encoding="utf-8") as f:
    json.dump(json_output, f, indent=2, ensure_ascii=False)

print(f"Extracted {len(json_output)} documents to extracted_documents.json")