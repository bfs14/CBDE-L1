import time, statistics, psycopg2, json
from config import load_config
import torch

def main():
    config = load_config()
    conn = psycopg2.connect(**config)
    cur = conn.cursor()

    # we get all the embeddings and sentences because we need to compare the 10 selected with every other embedding and calculate the distance
    cur.execute(""" SELECT s.id, s.text, e.embedding
       FROM bookcorpus s
       JOIN sentence_embeddings e ON s.id = e.sid
       ORDER BY s.id;"""
    )

    rows = cur.fetchall()
    cur.close()
    conn.close()

    ids = [row[0] for row in rows]
    texts = [row[1] for row in rows]
    vecs = [row[2] for row in rows]

    #convert to torch tensors
    allVecs = torch.tensor(vecs)

    #selecting first 10 sentences to compare with all embeddings
    selectedIds = ids[:10]
    selectedTexts = texts[:10]
    selectedVecs = allVecs[:10]

    cosineTimes=[]
    euclideanTimes=[]

    for qid, qvec in zip(selectedIds, selectedVecs):
        # cosine
        startTime = time.perf_counter()
        cosineScores = torch.matmul(allVecs, qvec)
        topVals, topIdxs = torch.topk(cosineScores, k=3)
        endTime = time.perf_counter()
        cosineTimes.append(endTime - startTime)

        # euclidean
        startTime = time.perf_counter()
        eucDist = torch.norm(allVecs - qvec, dim=1)
        topVals, topIdxs = torch.topk(-eucDist, k=3)
        endTime = time.perf_counter()
        euclideanTimes.append(endTime - startTime)

    #cosine results
    mn = min(cosineTimes)*1000
    mx = max(cosineTimes)*1000
    mean = statistics.mean(cosineTimes)*1000
    standardDeviation = statistics.pstdev(cosineTimes)*1000 if len(cosineTimes) > 1 else 0.0
    
    print("\nCosine query times (ms)")
    print(f"Min: {mn:.3f} ms")
    print(f"Max: {mx:.3f} ms")
    print(f"Average: {mean:.3f} ms")
    print(f"Standard Deviation: {standardDeviation:.3f} ms")

    #Euclidean results 
    mn = min(euclideanTimes)*1000
    mx = max(euclideanTimes)*1000
    mean = statistics.mean(euclideanTimes)*1000
    standardDeviation = statistics.pstdev(euclideanTimes)*1000 if len(euclideanTimes) > 1 else 0.0
    
    print("\nEuclidean query times (ms)")
    print(f"Min: {mn:.3f} ms")
    print(f"Max: {mx:.3f} ms")
    print(f"Average: {mean:.3f} ms")
    print(f"Standard Deviation: {standardDeviation:.3f} ms")

if __name__ == '__main__':
    main()
