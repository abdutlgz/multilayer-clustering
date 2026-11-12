"""Algorithm 1, all-layer spectral clustering and the UASE-based method (Section 5.2 of the paper).

All ranks are K = 5. Every eigenvector computation and every k-means call has a fixed seed,
so the results do not depend on the order of the computations.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, eigsh

K = 5                       # rank of every layer subspace and of every group subspace
EIG_TOL = 1e-7
EIG_MAXITER = 5000
KMEANS_RESTARTS = 40
KMEANS_MAXITER = 1000


# ---------------------------------------------------------------- building blocks

def hollowed_space(a, k, seed):
    """k leading eigenvectors (largest |eigenvalue|) of the hollowed Gram matrix A A^T - diag(A A^T).

    The n x n matrix is never formed; eigsh uses matrix-vector products.
    """
    degree = np.asarray(a.multiply(a).sum(1)).ravel()

    def multiply(x):
        return a @ (a.T @ x) - (degree * x if x.ndim == 1 else degree[:, None] * x)

    op = LinearOperator((a.shape[0], a.shape[0]), matvec=multiply, matmat=multiply, dtype=np.float64)
    values, u = eigsh(op, k=k, which='LM', tol=EIG_TOL, maxiter=EIG_MAXITER,
                      v0=np.random.default_rng(seed).normal(size=a.shape[0]))
    order = np.argsort(-np.abs(values))
    values, u = values[order], u[:, order]
    residual = np.linalg.norm(multiply(u) - u * values, axis=0) / np.maximum(abs(values), 1)
    assert max(residual) < 1e-5 and np.linalg.norm(u.T @ u - np.eye(k)) < 1e-5
    return u


def common_space(a, k, seed):
    """k leading left singular vectors of a (eigenvectors of A A^T), used by the UASE-based method."""
    def product(x):
        return a @ (a.T @ x)

    op = LinearOperator((a.shape[0], a.shape[0]), matvec=product, matmat=product, dtype=np.float64)
    values, u = eigsh(op, k=k, which='LA', tol=EIG_TOL, maxiter=EIG_MAXITER,
                      v0=np.random.default_rng(seed).normal(size=a.shape[0]))
    order = np.argsort(-values)
    values, u = values[order], u[:, order]
    residual = np.linalg.norm(product(u) - u * values, axis=0) / np.maximum(abs(values), 1)
    assert min(values) > 0 and max(residual) < 1e-5 and np.linalg.norm(u.T @ u - np.eye(k)) < 1e-5
    return u


def projected_hollowed(a, u):
    """U^T (A A^T - diag(A A^T)) U."""
    f = a.T @ u
    degree = np.asarray(a.multiply(a).sum(1)).ravel()
    b = f.T @ f - u.T @ (degree[:, None] * u)
    return (b + b.T) / 2


def kmeans(x, m, seed):
    """k-means with k-means++ starts; the best of KMEANS_RESTARTS runs (smallest within-group sum of squares)."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(KMEANS_RESTARTS):
        centers = [x[rng.integers(len(x))].copy()]
        while len(centers) < m:
            dd = np.min(((x[:, None] - np.array(centers)[None]) ** 2).sum(2), axis=1)
            if dd.sum() == 0:
                break
            centers.append(x[rng.choice(len(x), p=dd / dd.sum())].copy())
        if len(centers) < m:
            continue
        centers = np.array(centers)
        previous = None
        for _ in range(KMEANS_MAXITER):
            lab = ((x[:, None] - centers[None]) ** 2).sum(2).argmin(1)
            if len(np.unique(lab)) < m:
                break
            if previous is not None and np.array_equal(lab, previous):
                loss = float(((x - centers[lab]) ** 2).sum())
                if best is None or loss < best[0]:
                    best = (loss, lab.copy())
                break
            centers = np.array([x[lab == j].mean(0) for j in range(m)])
            previous = lab.copy()
    if best is None:
        raise RuntimeError('All k-means restarts failed or collapsed')
    return best[1]


def leading_eigenvectors(g, m):
    """m leading eigenvectors of the symmetric matrix g (the spectral embedding of the layers)."""
    v, w = np.linalg.eigh((g + g.T) / 2)
    return w[:, np.argsort(v)[-m:]]


def ari(a, b):
    """Adjusted Rand index between two labelings."""
    _, a = np.unique(a, return_inverse=True)
    _, b = np.unique(b, return_inverse=True)
    t = np.zeros((a.max() + 1, b.max() + 1), int)
    np.add.at(t, (a, b), 1)
    choose = lambda x: np.sum(x * (x - 1) / 2)
    total = len(a) * (len(a) - 1) / 2
    expected = choose(t.sum(0)) * choose(t.sum(1)) / total
    denom = (choose(t.sum(0)) + choose(t.sum(1))) / 2 - expected
    return float((choose(t) - expected) / denom) if denom else 1.


# ---------------------------------------------------------------- active layers

def active_rule(mats):
    """The rule of Section 4: layer l is active if n_l >= log n / rho_hat (beta_l >= beta_star).

    rho_hat is the fraction of entries equal to one over all layers. Returns (active flags, rho_hat, threshold).
    """
    n = mats[0].shape[0]
    rho_hat = sum(a.nnz for a in mats) / (n * sum(a.shape[1] for a in mats))
    threshold = np.log(n) / rho_hat
    return [a.shape[1] >= threshold for a in mats], rho_hat, threshold


# ---------------------------------------------------------------- the three methods

def fit(mats, m, seed):
    """Group the layers `mats` (list of sparse n x n_l matrices) into m groups with all three methods.

    Returns (labels, info): labels[method] is a list with one group label per layer, for
    method in 'algorithm1', 'all_layer', 'uase'; info has the active flags, rho_hat, the threshold
    and the similarity matrix G(i, j) = ||U_i^T U_j||_F^2 of all layers.
    """
    active, rho_hat, threshold = active_rule(mats)
    # Step 1a: subspace of every layer (seed 10000 + layer index)
    u = [hollowed_space(a, K, 10000 + i) for i, a in enumerate(mats)]
    g = np.array([[np.linalg.norm(x.T @ y, 'fro') ** 2 for y in u] for x in u])
    act = [i for i, f in enumerate(active) if f]
    labels = {}

    # Algorithm 1. Steps 1b-1c: spectral clustering of the active layers.
    lab = kmeans(leading_eigenvectors(g[np.ix_(act, act)], m), m, seed)
    alg1 = {i: int(l) for i, l in zip(act, lab)}
    # Steps 2a-2b: subspace of each group from its active layers (seed 300000 + group).
    group_space = [hollowed_space(sparse.hstack([mats[i] for i, l in zip(act, lab) if l == k], format='csc'),
                                  K, 300000 + k) for k in range(m)]
    # Step 2c: each non-active layer goes to the group whose subspace captures the largest part of it.
    for i, a in enumerate(mats):
        if not active[i]:
            alg1[i] = int(np.argmax([np.linalg.norm(a.T @ v, 'fro') ** 2 for v in group_space]))
    labels['algorithm1'] = [alg1[i] for i in range(len(mats))]

    # All-layer spectral clustering: Steps 1a-1c on all layers.
    labels['all_layer'] = [int(l) for l in kmeans(leading_eigenvectors(g, m), m, seed)]

    # UASE-based method: common subspace of dimension K*m from all layers, projected hollowed
    # matrices as features (unit norm), m leading left singular vectors with unit rows, k-means.
    v = common_space(sparse.hstack(mats, format='csc'), K * m, 710000)
    x = []
    for a in mats:
        f = projected_hollowed(a, v).ravel(order='F')
        norm = np.linalg.norm(f)
        x.append(f / norm if norm > 0 else f)
    w = np.linalg.svd(np.array(x), full_matrices=False)[0][:, :m]
    w = w / np.maximum(np.linalg.norm(w, axis=1)[:, None], 1e-12)
    labels['uase'] = [int(l) for l in kmeans(w, m, seed)]

    return labels, {'active': [bool(f) for f in active], 'rho_hat': rho_hat, 'threshold': threshold, 'similarity': g}
