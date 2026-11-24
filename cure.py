"""
Every point starts as its own cluster and the two closest clusters are merged until k are left.
A cluster is represented by a few scattered points that are moved (shrunk) towards the cluster mean,
and the distance between two clusters is the distance between their closest representatives.
Used by 02_cure, 06, 07 and 08.
"""
import numpy as np


def pairwise_squared_distances(points, block=2_000):
    """Squared distance between all pairs of points. The diagonal is inf so a point is never its own closest."""
    n = len(points)
    norms = (points ** 2).sum(axis=1)
    dist = np.zeros((n, n), dtype=np.float32)
    for start in range(0, n, block):
        p = points[start:start + block]
        dist[start:start + block] = np.maximum(norms[start:start + block, None] + norms[None, :] - 2 * p @ points.T, 0)
    np.fill_diagonal(dist, np.inf)
    return dist


def scattered_points(points, n, mean):
    """n points that are spread out: first the one farthest from the mean, then each time the point
    farthest from the ones already picked."""
    if len(points) <= n:
        return points.copy()
    picked = [np.argmax(((points - mean) ** 2).sum(axis=1))]
    dist_to_picked = ((points - points[picked[0]]) ** 2).sum(axis=1)
    for _ in range(n - 1):
        picked.append(np.argmax(dist_to_picked))
        dist_to_picked = np.minimum(dist_to_picked, ((points - points[picked[-1]]) ** 2).sum(axis=1))
    return points[picked].copy()


def cure(points, k_values, n_representatives=8, shrink=0.8, min_cluster_size=10):
    """
    n_representatives: representative points per cluster
    shrink: 0 = no shrinking, 1 = all representatives at the cluster mean
    min_cluster_size: clusters with fewer points are dropped as outliers
    Returns {k: (representative points, cluster number of each representative)} for every k in k_values.
    """
    n, dim = points.shape
    c = n_representatives

    # every point starts as its own cluster
    members = [[i] for i in range(n)]
    scattered = [points[i:i + 1] for i in range(n)]
    sums = points.copy()
    sizes = np.ones(n, dtype=int)
    alive = np.ones(n, dtype=bool)

    # representatives of every cluster; an unused slot has norm inf so it is never the closest
    reps = np.zeros((n, c, dim))
    reps[:, 0] = points
    rep_norms = np.full((n, c), np.inf)
    rep_norms[:, 0] = (points ** 2).sum(axis=1)

    # distances between clusters, and for every cluster the closest other cluster
    dist = pairwise_squared_distances(points)
    closest = dist.argmin(axis=1)
    closest_dist = dist[np.arange(n), closest].astype(float)

    def update_closest(clusters):
        for j in clusters:
            closest[j] = np.argmin(dist[j])
            closest_dist[j] = dist[j, closest[j]]

    def delete(clusters):
        alive[clusters] = False
        dist[clusters, :] = np.inf
        dist[:, clusters] = np.inf
        closest_dist[clusters] = np.inf
        update_closest(np.flatnonzero(alive & np.isin(closest, clusters)))

    result = {}
    outliers_removed = False
    while alive.sum() > min(k_values):
        # outliers: when 10% of the clusters are left, delete the clusters that are still one point
        if not outliers_removed and alive.sum() <= 0.1 * n:
            singletons = np.flatnonzero(alive & (sizes == 1))
            if len(singletons) < alive.sum() - min(k_values):
                delete(singletons)
            outliers_removed = True

        # merge the two closest clusters: b goes into a
        a = int(np.argmin(closest_dist))
        b = int(closest[a])
        members[a] += members[b]
        sums[a] += sums[b]
        sizes[a] += sizes[b]
        mean = sums[a] / sizes[a]

        # new representatives: scattered points of the merged cluster, shrunk towards the mean
        candidates = points[members[a]] if sizes[a] <= 50 else np.vstack([scattered[a], scattered[b]])
        scattered[a] = scattered_points(candidates, c, mean)
        new_reps = scattered[a] + shrink * (mean - scattered[a])
        delete([b])
        reps[a] = 0
        rep_norms[a] = np.inf
        reps[a, :len(new_reps)] = new_reps
        rep_norms[a, :len(new_reps)] = (new_reps ** 2).sum(axis=1)

        # distance from the merged cluster to every other cluster = closest pair of representatives
        others = np.flatnonzero(alive)
        others = others[others != a]
        d = (rep_norms[others][:, :, None] + (new_reps ** 2).sum(axis=1)[None, None, :]
             - 2 * reps[others] @ new_reps.T)                              # (others, c, new reps)
        new_dist = np.full(n, np.inf)
        new_dist[others] = np.maximum(d.min(axis=(1, 2)), 0)
        dist[a, :] = new_dist
        dist[:, a] = new_dist

        # update the closest clusters
        closer = new_dist < closest_dist
        closest[closer] = a
        closest_dist[closer] = new_dist[closer]
        update_closest(np.r_[np.flatnonzero(alive & (closest == a) & ~closer), a])

        # save the clustering when we reach one of the k values (small clusters are left out)
        for k in k_values:
            if alive.sum() <= k and k not in result:
                big = [i for i in np.flatnonzero(alive) if sizes[i] >= min_cluster_size]
                points_k = np.vstack([reps[i][np.isfinite(rep_norms[i])] for i in big])
                owner_k = np.concatenate([np.full(np.isfinite(rep_norms[i]).sum(), j) for j, i in enumerate(big)])
                result[k] = (points_k, owner_k)
    return result
