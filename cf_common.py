"""
Code shared by all the notebooks: loading the prepared data, the recommender, the metrics and saving results.
Every method notebook loads the data, makes cluster labels, and calls evaluate_clustering and save_results,
so all methods are evaluated the same way.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize

CACHE_DIR = Path("cache")
RESULTS_DIR = Path("results")

N_NEIGHBOURS = 50          # neighbours used by the recommender
TOP_KS = [5, 10, 20]       # K in Precision@K and Recall@K
BATCH_SIZE = 200           # users handled at once, only to save memory


class Data:
    """Rows are users, columns are items."""

    def __init__(self, ratings, embedding, eval_users, hidden_items):
        self.ratings = ratings.tocsr()            # log(1 + playcount), training rows only
        self.embedding = embedding                # 8-dim user vectors, used for clustering
        self.eval_users = eval_users              # the users we make recommendations for
        self.hidden_items = hidden_items          # their 10 hidden items (-1 = not in the vocabulary)

        self.user_vectors = normalize(self.ratings)                       # length 1, so dot product = cosine
        self.popularity = self.ratings.getnnz(axis=0).astype(np.float32)  # number of users per item
        self.n_users = self.ratings.shape[0]


def load_data():
    data = Data(sp.load_npz(CACHE_DIR / "ratings.npz"),
                np.load(CACHE_DIR / "embedding.npy"),
                np.load(CACHE_DIR / "eval_users.npy"),
                np.load(CACHE_DIR / "hidden_items.npy"))
    print(f"{data.n_users:,} users, {data.ratings.shape[1]:,} items, "
          f"{len(data.eval_users):,} users are evaluated")
    return data


def top_n(scores, n):
    """Column numbers of the n highest scores in every row, highest first."""
    best = np.argpartition(-scores, n, axis=1)[:, :n]
    order = np.argsort(-np.take_along_axis(scores, best, axis=1), axis=1)
    return np.take_along_axis(best, order, axis=1)


def recommend(data, labels, n_recs=max(TOP_KS)):
    """
    User-based kNN CF where the neighbours have to be in the same cluster.
    For every evaluated user: cosine similarity to everyone in their cluster, keep the 50 most similar,
    score of an item = sum of similarity * the neighbour's rating, recommend the best items the user doesn't have.
    Also returns some numbers about the neighbours.
    """
    recs = np.zeros((len(data.eval_users), n_recs), dtype=np.int64)
    n_similarities = 0
    n_neighbours = np.zeros(len(data.eval_users))
    mean_similarity = np.zeros(len(data.eval_users))
    tie_breaker = 1e-6 * data.popularity / data.popularity.max()   # equal scores: the more popular item wins

    eval_labels = labels[data.eval_users]
    for cluster in np.unique(eval_labels):
        members = np.flatnonzero(labels == cluster)                 # everybody in the cluster
        rows = np.flatnonzero(eval_labels == cluster)               # the evaluated users in it
        members_vectors_T = data.user_vectors[members].T.tocsr()
        members_ratings = data.ratings[members]

        for start in range(0, len(rows), BATCH_SIZE):
            batch_rows = rows[start:start + BATCH_SIZE]
            batch_users = data.eval_users[batch_rows]

            sim = (data.user_vectors[batch_users] @ members_vectors_T).toarray()
            n_similarities += sim.size

            # a user can't be their own neighbour
            own_column = np.searchsorted(members, batch_users)
            sim[np.arange(len(batch_users)), own_column] = 0

            # keep the 50 most similar users
            if sim.shape[1] > N_NEIGHBOURS:
                cutoff = np.partition(sim, -N_NEIGHBOURS, axis=1)[:, -N_NEIGHBOURS]
                sim[sim < cutoff[:, None]] = 0

            used = (sim > 0).sum(axis=1)
            n_neighbours[batch_rows] = used
            mean_similarity[batch_rows] = sim.sum(axis=1) / np.maximum(used, 1)

            # users without neighbours only get the tie breaker, i.e. the most popular items
            scores = (sp.csr_matrix(sim) @ members_ratings).toarray() + tie_breaker
            scores[data.ratings[batch_users].nonzero()] = -np.inf      # not what they already have
            recs[batch_rows] = top_n(scores, n_recs)

    has_neighbours = n_neighbours > 0
    stats = {
        "similarities": n_similarities,
        "neighbours used": n_neighbours.mean(),
        "users without neighbours %": 100 * (~has_neighbours).mean(),
        "avg neighbour similarity": mean_similarity[has_neighbours].mean() if has_neighbours.any() else np.nan,
    }
    return recs, stats


def count_hits(data, recs):
    """hits[i, j] is True if recommendation j for user i is one of their hidden items."""
    hits = np.zeros(recs.shape, dtype=bool)
    for i in range(len(recs)):
        hits[i] = np.isin(recs[i], data.hidden_items[i])
    return hits


def precision_recall(hits):
    """Mean Precision@K and Recall@K over the users, with the half-width of a 95% confidence interval."""
    n_hidden = 10
    result = {}
    for k in TOP_KS:
        hits_at_k = hits[:, :k].sum(axis=1)
        for name, values in [(f"P@{k}", hits_at_k / k), (f"R@{k}", hits_at_k / n_hidden)]:
            result[name] = values.mean()
            result[name + " ci95"] = 1.96 * values.std(ddof=1) / np.sqrt(len(values))
    return result


def evaluate_clustering(data, method, k, labels, fit_seconds, extra=None):
    """Runs the recommender with the given cluster labels and returns a summary row and the hits per user.
    extra: more columns for the results table, e.g. {"sample size": 20000}."""
    start = time.perf_counter()
    recs, stats = recommend(data, labels)
    cf_seconds = time.perf_counter() - start

    hits = count_hits(data, recs)
    all_sizes = np.bincount(labels)
    sizes = all_sizes[all_sizes > 0]

    summary = {
        "method": method,
        "k": k,
        **(extra or {}),
        "clusters": len(sizes),
        "largest cluster %": 100 * sizes.max() / data.n_users,
        "avg neighbour pool": all_sizes[labels[data.eval_users]].mean(),   # size of the evaluated users' clusters
        **precision_recall(hits),
        "clustering s": fit_seconds,
        "CF s (eval users)": cf_seconds,
        "CF s (all users, estimated)": cf_seconds / len(data.eval_users) * data.n_users,
        "similarities (millions)": stats["similarities"] / 1e6,
        "neighbours used": stats["neighbours used"],
        "users without neighbours %": stats["users without neighbours %"],
        "avg neighbour similarity": stats["avg neighbour similarity"],
    }
    per_user = pd.DataFrame({
        "method": method, "k": k, **(extra or {}), "user": data.eval_users,
        **{f"hits@{kk}": hits[:, :kk].sum(axis=1) for kk in TOP_KS},
    })
    print(f"{method:<16} k={k:<4} P@10 = {summary['P@10']:.4f}   "
          f"clustering {fit_seconds:6.1f}s   CF {cf_seconds:6.1f}s")
    return summary, per_user


def save_results(name, summaries, per_user_tables, folder=RESULTS_DIR):
    """The main methods save to results/, the extra analyses to results/analyses/."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summaries)
    summary.to_csv(folder / f"{name}.csv", index=False)
    pd.concat(per_user_tables).to_csv(folder / f"{name}_per_user.csv", index=False)
    print(f"saved {folder / name}.csv and {folder / name}_per_user.csv")
    return summary


def assign_to_nearest(points, centers, batch=50_000):
    """Number of the nearest center (Euclidean distance) for every point."""
    labels = np.zeros(len(points), dtype=np.int64)
    center_norms = (centers ** 2).sum(axis=1)
    for start in range(0, len(points), batch):
        p = points[start:start + batch]
        dist = center_norms[None, :] - 2 * p @ centers.T     # |p|^2 is the same for every center, so it is left out
        labels[start:start + batch] = dist.argmin(axis=1)
    return labels
