"""
Neighbour recall: for every evaluated user, the share of their 50 most similar users (out of all users)
that are in the same cluster. 100% means no clustering. It checks a clustering without the recommender.
"""
from pathlib import Path

import numpy as np

from cf_common import BATCH_SIZE, N_NEIGHBOURS


def true_neighbours(data, cache_file="cache/true_neighbours.npy"):
    """The 50 most similar users (out of all users) of every evaluated user.
    Takes as long as a run without clustering, so it is saved in cache/."""
    path = Path(cache_file)
    if path.exists():
        return np.load(path)

    all_vectors_T = data.user_vectors.T.tocsr()
    result = np.zeros((len(data.eval_users), N_NEIGHBOURS), dtype=np.int64)
    for start in range(0, len(data.eval_users), BATCH_SIZE):
        users = data.eval_users[start:start + BATCH_SIZE]
        sim = (data.user_vectors[users] @ all_vectors_T).toarray()      # cosine similarity to everyone
        sim[np.arange(len(users)), users] = -1                          # not their own neighbour
        result[start:start + len(users)] = np.argpartition(-sim, N_NEIGHBOURS, axis=1)[:, :N_NEIGHBOURS]
    np.save(path, result)
    return result


def neighbour_recall(data, labels, neighbours):
    """Share of the true neighbours that are in the same cluster, averaged over the evaluated users."""
    same_cluster = labels[neighbours] == labels[data.eval_users][:, None]
    return same_cluster.mean()
