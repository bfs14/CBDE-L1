import time, statistics, psycopg2, json
from config import load_config
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
CSV_PATH = "bookcorpus10k.csv"
BATCH_SIZE = 256
DB_BATCH_SIZE = 200

DDL = """
    CREATE TABLE IF NOT EXISTS sentence_embeddings (
        sid INTEGER PRIMARY KEY,
        embedding JSONB NOT NULL
    );
"""

DB_INSERT_QUERY = """
    INSERT INTO sentence_embeddings (sid, embedding)
    VALUES %s
    ON CONFLICT (sid) DO UPDATE SET embedding = EXCLUDED.embedding;
"""

def main():
    config = load_config()
    conn = psycopg2.connect(**config)
    cur = conn.cursor()
    cur.execute(DDL)
    conn.commit()

    cur.execute("SELECT id, text FROM bookcorpus ORDER BY id;")
    rows = cur.fetchall()
    sids = [row[0] for row in rows]
    sentences = [row[1] for row in rows]

    model = SentenceTransformer(MODEL_NAME)
    initTime = time.perf_counter()
    embeddings = model.encode(sentences, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True)
    endTime = time.perf_counter()

    print(f"\nTime taken to encode {len(sentences)} sentences: {endTime - initTime:.3f} seconds")

    calculateRowTimes = []
    batch = []

    for i, sid in enumerate(sids):
        embedding = embeddings[i].tolist()
        batch.append((int(sid), json.dumps(embedding)))

        if len(batch) >= DB_BATCH_SIZE:
            startTime = time.perf_counter()
            execute_values(cur, DB_INSERT_QUERY, batch,page_size=len(batch))
            conn.commit()
            endTime = time.perf_counter()
            perRowTime = (endTime - startTime)/len(batch)
            calculateRowTimes.extend([perRowTime]*len(batch))
            batch = []
    
    #insert if there are any other rows remaining
    if batch:
        startTime = time.perf_counter()
        execute_values(cur, DB_INSERT_QUERY, batch, page_size=len(batch))
        conn.commit()
        endTime = time.perf_counter()
        perRowTime = (endTime - startTime)/len(batch)
        calculateRowTimes.extend([perRowTime]*len(batch))
    
    cur.close()
    conn.close()

    if calculateRowTimes:
        mn = min(calculateRowTimes)*1000
        mx = max(calculateRowTimes)*1000
        mean = statistics.mean(calculateRowTimes)*1000
        standardDeviation = statistics.pstdev(calculateRowTimes)*1000 if len(calculateRowTimes) > 1 else 0.0

        print("\n Insertion times (JSONB)")
        print(f"Min: {mn:.3f} ms")
        print(f"Max: {mx:.3f} ms")
        print(f"Average: {mean:.3f} ms")
        print(f"Standard Deviation: {standardDeviation:.3f} ms")
    
    else:
        print("No rows were inserted.")

if __name__ == '__main__':
    main()
