# Multi-view clustering for collaborative filtering (Last.fm)
Everything runs on all 476,451 users. The main seed is 579.

## Data

Kaggle: "Music Listening Data (500k users)",
https://www.kaggle.com/datasets/gabrielkahen/music-listening-data-500k-users

Download the data from Kaggle into `data/`:

| file | rows | used for |
|---|---|---|
| `user_top_tracks.csv` | 23,822,550 | everything (recommended and evaluated view) |
| `user_top_artists.csv` | 23,822,550 | multi-view only |
| `user_top_albums.csv` | 23,822,550 | multi-view only |
| `users.csv` | 476,451 | not used (country and total scrobbles) |


## Setup and running

```
pip install -r requirements.txt
```

Tested with Python 3.12 (Anaconda) on Windows, 16 GB RAM. Run the notebooks from `tracks/` (e.g. open
`tracks/02_cure.ipynb` in Jupyter and "Run all"), in this order:

| notebook | what it does | time (full data) |
|---|---|---|
| `tracks/00_prepare_data` | split, matrix, embedding, test users -> `tracks/cache/` | 2 min |
| `tracks/01_no_clustering` | reference: kNN CF over all users | 1.5 min |
| `tracks/02_cure` | CURE (main method) | 2.5 min |
| `tracks/03_kmeans` | k-means | 1.5 min |
| `tracks/04_popularity` | baseline: most popular tracks | 0.5 min |
| `tracks/05_compare_results` | tables and plots from `tracks/results/` | seconds |
| `tracks/06_vocabulary_threshold` | CURE with a vocabulary of 20, 10 or 5 users per track | 12 min |
| `tracks/07_cure_sample_size` | CURE with a sample of 20,000 and 30,000 users | 20 min |
| `tracks/08_multiview` | multi-view CURE and k-means | 15-20 min |
| `tracks/09_view_clusters` | labelled cluster maps for the report (needs 02 and 08 first) | seconds |

Every notebook already contains its outputs, and `tracks/cache/`, `tracks/results/` and `tracks/figures/` are
included, so nothing has to be run to see the results.

## Folders

- `cf_common.py`: loading the cache, the recommender, the metrics and saving results (shared by all notebooks)
- `cure.py`: CURE, our own implementation
- `neighbours.py`: the true 50 nearest neighbours and "neighbour recall"
- `tracks/`: all notebooks, `cache/` (prepared data), `results/` (one CSV per method, `analyses/` for 06,
  07 and 08), `figures/` (report figures from 09)
- `data/`: the raw CSV files