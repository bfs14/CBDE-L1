import csv, re, time, statistics, uuid
from chromadb import Client
from chromadb.utils import embedding_functions

CSV_PATH = "bookcorpus10k.csv"
SPLITTED_SENTENCES = re.compile(r'(?<=[.!?])\s+')

class DummyEmbedding:
    def __call__(self, input):
        return [[0.0]*384 for _ in input]

    def name(self):
        return "dummy"

def main():
    client = Client()
    collection = client.get_or_create_collection(name="bookcorpus", embedding_function=DummyEmbedding())

    calculateRowTimes = []
    batchSize = 500
    docsBatch = []
    idsBatch = []

    with open(CSV_PATH, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)

        if 'text' not in reader.fieldnames:
            raise ValueError(f"Error in  {CSV_PATH}")

        for row in reader:
            for sentence in SPLITTED_SENTENCES.split(row.get('text', '') or ''):
                s = sentence.strip()

                if s:
                    docsBatch.append(s)
                    idsBatch.append(str(uuid.uuid4()))

                    if len(docsBatch) >= batchSize:
                        initTime = time.perf_counter()
                        collection.add(documents=docsBatch, ids=idsBatch)
                        endTime = time.perf_counter()
                        calculateRowTimes.append((endTime - initTime)/len(docsBatch))
                        docsBatch = []
                        idsBatch = []

    if docsBatch:
        initTime = time.perf_counter()
        collection.add(documents=docsBatch, ids=idsBatch)
        endTime = time.perf_counter()
        calculateRowTimes.append((endTime - initTime)/len(docsBatch))
    
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
        print("No data.")

if __name__ == '__main__':
    main()
