import csv, re, psycopg2, time, statistics
from config import load_config

CSV_PATH = "bookcorpus10k.csv"
SPLITTED_SENTENCES = re.compile(r'(?<=[.!?])\s+')

DDL = """
    CREATE TABLE IF NOT EXISTS bookcorpus (
        id SERIAL PRIMARY KEY,
        text TEXT NOT NULL
    );
"""

def main():
    config = load_config()
    conn = psycopg2.connect(**config)
    cur = conn.cursor()
    cur.execute(DDL)
    conn.commit()

    calculateRowTimes = []

    with open(CSV_PATH, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)

        if 'text' not in reader.fieldnames:
            raise ValueError(f"Error in  {CSV_PATH}")

        for row in reader:
            for sentence in SPLITTED_SENTENCES.split(row.get('text', '') or ''):
                s = sentence.strip()

                if s:
                    initTime = time.perf_counter()
                    cur.execute("INSERT INTO bookcorpus (text) VALUES (%s)", (s,))
                    endTime = time.perf_counter()
                    calculateRowTimes.append(endTime - initTime)
    
    conn.commit()
    cur.close()
    conn.close()

    if calculateRowTimes:
        mn = min(calculateRowTimes)*1000
        mx = max(calculateRowTimes)*1000
        mean = statistics.mean(calculateRowTimes)*1000

        if len(calculateRowTimes) > 1:
            standardDeviation = statistics.pstdev(calculateRowTimes)*1000
        
        else:
            standardDeviation = 0.0

        print("\nInsertion times (TEXT)")
        print(f"Min: {mn:.3f} ms")
        print(f"Max: {mx:.3f} ms")
        print(f"Average: {mean:.3f} ms")
        print(f"Standard Deviation: {standardDeviation:.3f} ms")

    else:
        print("No rows were inserted.")
    
if __name__ == '__main__':
    main()
