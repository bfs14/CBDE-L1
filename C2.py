import time, statistics
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

def main():
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_collection(name="bookcorpus")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    docs = collection.get(include=["documents"])
    ids = docs["ids"]
    texts = docs["documents"]

    queryIds = ids[:10]
    queryTexts = texts[:10]

    cosTimes = []
    eucTimes = []

    for qid, qtext in zip(queryIds, queryTexts):
        qemb = model.encode([qtext])[0]

        # Cosine
        start = time.perf_counter()
        res = collection.query(query_embeddings=[qemb], n_results=2, include=["documents","distances"])
        end = time.perf_counter()
        cosTimes.append(end - start)

        print(f"\nQuery: {qtext[:50]}...")
        print("Cosine top2:", list(zip(res["ids"][0], res["documents"][0])))

        # Euclidean
        allEmbs = collection.get(include=["embeddings"])["embeddings"]
        import numpy as np
        arr = np.array(allEmbs)
        dists = np.linalg.norm(arr - qemb, axis=1)
        top2Idx = dists.argsort()[:2]
        end = time.perf_counter()
        eucTimes.append(end - start)

        print("Euclidean top2:", [texts[i][:50] for i in top2Idx])

    # métricas
    print("\nCosine query times (ms)")
    print_stats(cosTimes)
    print("\nEuclidean query times (ms)")
    print_stats(eucTimes)

def print_stats(times):
    mn = min(times)*1000
    mx = max(times)*1000
    mean = statistics.mean(times)*1000
    std = statistics.pstdev(times)*1000 if len(times) > 1 else 0.0
    print(f"Min: {mn:.3f} ms")
    print(f"Max: {mx:.3f} ms")
    print(f"Average: {mean:.3f} ms")
    print(f"StdDev: {std:.3f} ms")

if __name__ == "__main__":
    main()
