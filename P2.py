import time, statistics, psycopg2, json
from config import load_config
import torch

def main():
    config = load_config()
    conn = psycopg2.connect(**config)
    cur = conn.cursor()

    # obtener todas las oraciones con sus embeddings
    cur.execute("""
        SELECT s.id, s.text, e.embedding
        FROM bookcorpus s
        JOIN sentence_embeddings e ON s.id = e.sid
        ORDER BY s.id;
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    ids = [row[0] for row in rows]
    texts = [row[1] for row in rows]
    vecs = [json.loads(row[2]) if isinstance(row[2], str) else row[2] for row in rows]

    allVecs = torch.tensor(vecs)

    # seleccionamos las primeras 10
    selectedIds = ids[:10]
    selectedTexts = texts[:10]
    selectedVecs = allVecs[:10]

    cosineTimes = []
    euclideanTimes = []

    for qid, qtext, qvec in zip(selectedIds, selectedTexts, selectedVecs):
        # Cosine
        startTime = time.perf_counter()
        cosineScores = torch.matmul(allVecs, qvec)
        topVals, topIdxs = torch.topk(cosineScores, k=3)
        endTime = time.perf_counter()
        cosineTimes.append(endTime - startTime)

        # Euclidean
        startTime = time.perf_counter()
        eucDist = torch.norm(allVecs - qvec, dim=1)
        topValsE, topIdxsE = torch.topk(-eucDist, k=3)
        endTime = time.perf_counter()
        euclideanTimes.append(endTime - startTime)

        # Mostrar resultados (excluyendo la query misma)
        print(f"\nQuery: {qtext[:50]}...")
        print("Cosine top2:", [(ids[i], texts[i][:80]) for i in topIdxs if ids[i] != qid][:2])
        print("Euclidean top2:", [texts[i][:80] for i in topIdxsE if ids[i] != qid][:2])

    # Métricas Cosine
    print("\nCosine query times (ms)")
    print_stats(cosineTimes)

    # Métricas Euclidean
    print("\nEuclidean query times (ms)")
    print_stats(euclideanTimes)

def print_stats(times):
    mn = min(times) * 1000
    mx = max(times) * 1000
    mean = statistics.mean(times) * 1000
    std = statistics.pstdev(times) * 1000 if len(times) > 1 else 0.0
    print(f"Min: {mn:.3f} ms")
    print(f"Max: {mx:.3f} ms")
    print(f"Average: {mean:.3f} ms")
    print(f"Standard Deviation: {std:.3f} ms")

if __name__ == '__main__':
    main()
