from datasets import load_dataset
import pandas as pd

# Carregar el dataset bookcorpus
dataset = load_dataset("rojagtap/bookcorpus", split="train")

# Seleccionar les primeres 10k files
subset = dataset.select(range(10000))

# Guardar el subset a un arxiu CSV
subset.to_csv("bookcorpus10k.csv", index=False)
