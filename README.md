# RecSys 02 – Transformer-Based Sequential Recommender

This project implements a Transformer-based sequential recommendation model inspired by SASRec and BERT4Rec.  
The model predicts the next item a user will interact with based on their historical sequence of items.

## Architecture Overview

The system follows a standard Transformer encoder architecture:

```
User Sequence  --->  Item Embedding  --->  Positional Encoding  --->  Transformer Encoder  --->  Next‑Item Prediction
                               (IDs)                 (sin/cos)              (multi‑head attention)      (softmax over items)
```


## Dataset

This project uses the MovieLens 20M dataset from Kaggle.

Required file:

```
data/raw/movielens-20m/ratings.csv
```

Expected columns (MovieLens standard):

```
userId, movieId, rating, timestamp
```

## Preprocessing

Sequences are generated using:

```
python -m recsys_transformer_seq.data.movielens_preprocess
```

This produces:

```
data/processed/movielens_sequences.csv
```

Each row contains:

- userId  
- sequence: a space‑separated list of movieIds  
- target: the next movieId to predict  

## Training

Run training with:

```
python -m recsys_transformer_seq.train_seq
```

This saves the trained model to:

```
data/models/transformer_seq_movielens.pt
```

Device selection:

- CUDA if available  
- Otherwise Apple MPS  
- Otherwise CPU  

## Testing in Notebook

Open:

```
notebooks/01_movielens_inference.ipynb
```

The notebook demonstrates:

1. Loading the preprocessed sequences  
2. Loading the trained model checkpoint  
3. Running next‑item prediction  
4. Getting Top‑K recommended items  

## Project Structure

```
recsys-02-transformer-seq/
├── config/
│   └── config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
├── notebooks/
│   └── 01_movielens_inference.ipynb
├── src/recsys_transformer_seq/
│   ├── config.py
│   ├── train_seq.py
│   ├── data/
│   │   ├── movielens_preprocess.py
│   │   └── seq_dataset.py
│   └── models/
│       └── transformer_seq.py
└── tests/
    ├── test_smoke.py
    └── test_transformer_shapes.py
```

## Goal

This repository serves as a clean, minimal implementation of a modern sequential recommendation pipeline that is easy to train, evaluate, and extend.
