import csv, time, statistics, uuid
import chromadb
from chromadb.utils import embedding_functions

CSV_PATH = "bookcorpus10k.csv"

def main():
    client = chromadb.PersistentClient(path="chroma_db")

    embeddingFunction = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="sentence-transformers/all-MiniLM-L6-v2")
    collection = client.get_or_create_collection(name="bookcorpus", embedding_function=embeddingFunction)

    ids = []
    docs = []

    with open(CSV_PATH, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        
        if 'text' not in reader.fieldnames:
            raise ValueError(f"Error in  {CSV_PATH}")
        
        for i, row in enumerate(reader):
            txt = (row.get('text', '') or '').strip()
            
            if txt:
                ids.append(str(uuid.uuid4()))
                docs.append(txt)

    calculateRowTimes = []
    batchSize = 200
    
    for i in range(0, len(docs), batchSize):
        batchDocs = docs[i:i+batchSize]
        batchIds = ids[i:i+batchSize]
        initTime = time.perf_counter()
        collection.add(documents=batchDocs, ids=batchIds)
        endTime = time.perf_counter()
        perRowTime = (endTime - initTime)/len(batchDocs)
        calculateRowTimes.extend([perRowTime]*len(batchDocs))

    #results
    mn = min(calculateRowTimes)*1000
    mx = max(calculateRowTimes)*1000
    mean = statistics.mean(calculateRowTimes)*1000
    standardDeviation = statistics.pstdev(calculateRowTimes)*1000 if len(calculateRowTimes) > 1 else 0.0

    print("\n Insertion times (ChromaDB)")
    print(f"Min: {mn:.3f} ms")
    print(f"Max: {mx:.3f} ms")
    print(f"Average: {mean:.3f} ms")
    print(f"Standard Deviation: {standardDeviation:.3f} ms")

if __name__ == '__main__':
    main()
