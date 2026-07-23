#!/usr/bin/env python3
"""Core routines for HHIN reduction and semantic-preservation evaluation."""
from __future__ import annotations

import argparse
import collections
import json
import os
import time
import tracemalloc
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from sklearn.preprocessing import normalize

CONFIGS = {
    'ACM': {
        'target_type': 0,
        'target_name': 'paper',
        'main_types': [0, 1, 2],     # P A S
        'semantic_types': [3],       # T
        'core_link_types': [2, 3, 4, 5],
        'term_paths_by_type': {0: [6], 1: [3, 6], 2: [5, 6]},
        'meta_paths': {
            'PAP': [2],
            'PSP': [4],
            'PTP': [6],
        },
        'strong_attr_types': [0],
    },
    'DBLP': {
        'target_type': 0,
        'target_name': 'author',
        'main_types': [0, 1, 3],     # A P V
        'semantic_types': [2],       # T
        'core_link_types': [0, 2, 3, 5],
        'term_paths_by_type': {0: [0, 1], 1: [1], 3: [5, 1]},
        'meta_paths': {
            'APA': [0],
            'APVPA': [0, 2],
            'APTPA': [0, 1],
        },
        'strong_attr_types': [0],
    },
    'IMDB': {
        'target_type': 0,
        'target_name': 'movie',
        'main_types': [0, 1, 2],     # M D A
        'semantic_types': [3],       # K
        'core_link_types': [0, 1, 2, 3],
        'term_paths_by_type': {0: [4], 1: [1, 4], 2: [3, 4]},
        'meta_paths': {
            'MDM': [0],
            'MAM': [2],
            'MKM': [4],
        },
        'strong_attr_types': [0],
    },
}

TYPE_NAME_CACHE: Dict[str, Dict[int, str]] = {}


def clean_names(z: zipfile.ZipFile) -> List[str]:
    return [n for n in z.namelist() if not n.startswith('__MACOSX') and '/._' not in n and '.DS_Store' not in n]


def load_hgb_zip(path: str):
    z = zipfile.ZipFile(path)
    names = clean_names(z)
    prefix = ''
    for n in names:
        if n.endswith('node.dat'):
            prefix = n.rsplit('/', 1)[0] + '/' if '/' in n else ''
            break
    info = json.loads(z.read(prefix + 'info.dat').decode('utf-8'))
    type_map = {int(k): v for k, v in info['node.dat']['node type'].items()}
    TYPE_NAME_CACHE[path] = type_map
    dataset_key = os.path.basename(path).replace('.zip', '').upper()
    target_attr_types = set(CONFIGS.get(dataset_key, {}).get('strong_attr_types', []))

    node_lines = z.read(prefix + 'node.dat').decode('utf-8').splitlines()
    node_records = []
    max_id = -1
    feat_store = {}
    type_counts = collections.Counter()
    for line in node_lines:
        if not line.strip():
            continue
        p = line.split('\t')
        nid = int(p[0])
        name = p[1] if len(p) > 1 else ''
        t = int(p[2])
        feat = p[3] if len(p) > 3 else ''
        node_records.append((nid, name, t, feat))
        max_id = max(max_id, nid)
        type_counts[t] += 1
        if t in target_attr_types and feat != '':
            feat_store[nid] = np.fromstring(feat, sep=',', dtype=np.float32)

    N = max_id + 1
    types = np.full(N, -1, dtype=np.int32)
    names_arr = np.empty(N, dtype=object)
    for nid, name, t, feat in node_records:
        types[nid] = t
        names_arr[nid] = name

    ids_by_type = {}
    feats_by_type = {}
    for t in sorted(type_counts):
        ids = np.where(types == t)[0]
        ids_by_type[t] = ids
        vals = []
        rows = []
        cols = []
        dim = 0
        for r, nid in enumerate(ids):
            arr = feat_store.get(nid)
            if arr is not None and len(arr) > 0:
                nz = np.nonzero(arr)[0]
                rows.extend([r] * len(nz))
                cols.extend(nz.tolist())
                vals.extend(arr[nz].tolist())
                dim = max(dim, len(arr))
        feats_by_type[t] = sp.csr_matrix((vals, (rows, cols)), shape=(len(ids), dim), dtype=np.float32) if dim > 0 else sp.csr_matrix((len(ids), 0), dtype=np.float32)

    edges_by_link = collections.defaultdict(list)
    link_defs = {}
    for lid, desc in info['link.dat']['link type'].items():
        link_defs[int(lid)] = (int(desc['start']), int(desc['end']), desc['meaning'])
    for line in z.read(prefix + 'link.dat').decode('utf-8').splitlines():
        if not line.strip():
            continue
        p = line.split('\t')
        s, d, lt = int(p[0]), int(p[1]), int(p[2])
        w = float(p[3]) if len(p) > 3 and p[3] else 1.0
        edges_by_link[lt].append((s, d, w))

    labels = []
    for fname in ['label.dat', 'label.dat.test']:
        labels_path = prefix + fname
        if labels_path in names:
            lab = []
            for line in z.read(labels_path).decode('utf-8').splitlines():
                if not line.strip():
                    continue
                p = line.split('\t')
                nid = int(p[0])
                t = int(p[-2])
                y = p[-1]
                lab.append((nid, t, y))
            labels.append((fname, lab))
    return {
        'path': path,
        'name': os.path.basename(path).replace('.zip', ''),
        'zip': z,
        'prefix': prefix,
        'info': info,
        'N': N,
        'types': types,
        'names': names_arr,
        'node_records': node_records,
        'type_counts': type_counts,
        'ids_by_type': ids_by_type,
        'feats_by_type': feats_by_type,
        'edges_by_link': edges_by_link,
        'link_defs': link_defs,
        'labels': dict(labels),
    }


def build_link_matrices(data):
    mats = {}
    maps = {t: {nid: i for i, nid in enumerate(data['ids_by_type'][t])} for t in data['ids_by_type']}
    for lt, edges in data['edges_by_link'].items():
        st, et, _ = data['link_defs'][lt]
        rows = []
        cols = []
        vals = []
        smap = maps[st]
        emap = maps[et]
        for s, d, w in edges:
            si = smap.get(s)
            di = emap.get(d)
            if si is not None and di is not None:
                rows.append(si)
                cols.append(di)
                vals.append(w)
        mats[lt] = sp.csr_matrix((vals, (rows, cols)), shape=(len(data['ids_by_type'][st]), len(data['ids_by_type'][et])), dtype=np.float32)
    return mats


def fixed_point_partition(data, mats, core_link_types, max_iter=20, use_entity_reference=True):
    """Compute stable main-structure candidate equivalence classes.

    This routine uses equivalence-class identifiers to describe the fixed-point
    refinement process on the HHIN main structure.

    Initial identifiers are determined by node types. In the (t+1)-th
    iteration, a node's new identifier is determined by its own type, adjacent
    relation types, neighbors' previous-round identifiers, and, by default,
    neighbors' entity reference labels. The iteration stops when all identifiers
    remain unchanged.

    Term/keyword semantic-channel matrices and node-attached attributes are not
    read or changed here. They remain handled by derive_term_matrix() and
    get_raw_attr_for_type() exactly as before.
    """
    part_local = {t: np.zeros(len(data['ids_by_type'][t]), dtype=np.int32) for t in data['ids_by_type']}
    types_sorted = sorted(data['ids_by_type'])
    entity_refs = {t: np.asarray(data['ids_by_type'][t], dtype=np.int64) for t in data['ids_by_type']}

    outgoing = defaultdict(list)
    for lt in core_link_types:
        st, et, _ = data['link_defs'][lt]
        outgoing[st].append((int(lt), et, mats[lt].tocsr()))

    def global_parts():
        gp = {}
        off = 0
        for t in types_sorted:
            gp[t] = part_local[t] + off
            off += int(part_local[t].max()) + 1 if len(part_local[t]) else 0
        return gp

    history = []
    for it in range(max_iter):
        gp = global_parts()
        new_part = {}
        num_changes = 0
        num_classes = 0
        for t in types_sorted:
            n = len(data['ids_by_type'][t])
            sig_to_id = {}
            ids = np.empty(n, dtype=np.int32)
            for i in range(n):
                sig_parts = []
                for lt, et, mat in outgoing.get(t, []):
                    row = mat.getrow(i)
                    cols = row.indices
                    if len(cols) == 0:
                        sig = (lt, ())
                    else:
                        neigh_parts = gp[et][cols]
                        if use_entity_reference:
                            neigh_refs = entity_refs[et][cols]
                            # Keep both the current block and entity identifiers.
                            records = sorted(
                                (int(block_id), int(entity_id))
                                for block_id, entity_id in zip(neigh_parts.tolist(), neigh_refs.tolist())
                            )
                            sig = (lt, tuple(records))
                        else:
                            uniq, cnts = np.unique(neigh_parts, return_counts=True)
                            sig = (lt, tuple(zip(uniq.tolist(), cnts.tolist())))
                    sig_parts.append(sig)
                # Retaining the previous identifier prevents block coarsening.
                full_sig = (int(t), int(part_local[t][i]), tuple(sig_parts))
                pid = sig_to_id.setdefault(full_sig, len(sig_to_id))
                ids[i] = pid
            new_part[t] = ids
            num_classes += len(np.unique(ids))
            if not np.array_equal(ids, part_local[t]):
                num_changes += int(np.sum(ids != part_local[t]))
        history.append((it + 1, num_classes, num_changes))
        if all(np.array_equal(new_part[t], part_local[t]) for t in types_sorted):
            return new_part, history
        part_local = new_part
    raise RuntimeError(
        f'Fixed-point refinement did not converge within {max_iter} iterations. '
        'Increase --max-iter only after checking the refinement signatures.'
    )

def compose_path(mats, link_seq):
    B = mats[link_seq[0]].tocsr().astype(np.float32)
    for lt in link_seq[1:]:
        B = (B @ mats[lt]).tocsr().astype(np.float32)
    return B


def derive_term_matrix(mats, link_seq):
    B = compose_path(mats, link_seq)
    if B.nnz > 0:
        B = B.copy().tocsr()
        B.data[:] = 1.0
        B.eliminate_zeros()
    return B


def get_raw_attr_for_type(data, t, strong_attr_types):
    return data['feats_by_type'][t].tocsr().astype(np.float32) if t in strong_attr_types else sp.csr_matrix((len(data['ids_by_type'][t]), 0), dtype=np.float32)


def exact_sparse_row_labels(X):
    """Label identical sparse rows while keeping empty rows as singletons."""
    X = X.tocsr()
    labels = np.empty(X.shape[0], dtype=np.int32)
    sig_to_id = {}
    next_id = 0
    for i in range(X.shape[0]):
        row = X.getrow(i)
        if row.nnz == 0:
            labels[i] = next_id
            next_id += 1
            continue
        sig = (tuple(row.indices.tolist()), tuple(np.asarray(row.data).tolist()))
        if sig not in sig_to_id:
            sig_to_id[sig] = next_id
            next_id += 1
        labels[i] = sig_to_id[sig]
    return labels


def cosine_matrix_sparse(X, dtype=np.float32):
    if X.shape[1] == 0:
        return None
    Xn = normalize(X.astype(dtype), norm='l2', axis=1, copy=True)
    sim = (Xn @ Xn.T).toarray().astype(dtype)
    return np.clip(sim, -1.0, 1.0)


def jaccard_matrix_binary(B):
    if B.shape[1] == 0:
        return None
    X = B.astype(np.float32)
    inter = (X @ X.T).toarray().astype(np.float32)
    deg = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
    union = deg[:, None] + deg[None, :] - inter
    sim = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    both_empty = (deg[:, None] == 0) & (deg[None, :] == 0)
    sim[both_empty] = 1.0
    return sim


def overlap_matrix_binary(B):
    """Return overlap coefficients for binary semantic-channel sets."""
    if B.shape[1] == 0:
        return None
    X = B.astype(np.float32)
    inter = (X @ X.T).toarray().astype(np.float32)
    deg = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
    denom = np.minimum(deg[:, None], deg[None, :])
    sim = np.divide(inter, denom, out=np.zeros_like(inter), where=denom > 0)
    both_empty = (deg[:, None] == 0) & (deg[None, :] == 0)
    sim[both_empty] = 1.0
    return sim


def term_similarity_matrix(T, mode='cosine'):
    """Compute cosine, Jaccard, or overlap similarity for a semantic channel."""
    mode = (mode or 'cosine').lower()
    if mode == 'cosine':
        return cosine_matrix_sparse(T.astype(np.float32))
    if mode == 'jaccard':
        return jaccard_matrix_binary(T)
    if mode == 'overlap':
        return overlap_matrix_binary(T)
    raise ValueError(f'Unknown --term-sim: {mode}. Choose from cosine, jaccard, overlap.')


def layered_similarity_matrices(X, T, term_sim='cosine'):
    """Return independent attribute and semantic-channel similarities."""
    # The attached-feature channel can use the strict tau_attr = 1 boundary;
    # double precision avoids splitting collinear rows because of roundoff.
    sim_attr = cosine_matrix_sparse(X, dtype=np.float64)
    sim_term = term_similarity_matrix(T, mode=term_sim)
    return sim_attr, sim_term


def _passes_layered_guard(cand: int, cluster: List[int], sim_attr, sim_term, tau_attr: float, tau_term: float) -> bool:
    if not cluster:
        return True
    if sim_attr is not None:
        if float(np.min(sim_attr[cand, cluster])) < tau_attr - 1e-10:
            return False
    if sim_term is not None:
        if float(np.min(sim_term[cand, cluster])) < tau_term - 1e-8:
            return False
    return True


def make_layered_threshold_mapping(parts_local, rawX, termB, tau_term, tau_attr, term_sim='cosine'):
    """Build a deterministic complete-link cover within each candidate class."""
    classes = defaultdict(list)
    for i, p in enumerate(parts_local):
        classes[int(p)].append(i)
    cluster_of = np.empty(len(parts_local), dtype=np.int32)
    clusters = []
    for _, idxs in classes.items():
        idxs = list(sorted(idxs))
        if len(idxs) == 1:
            cid = len(clusters)
            clusters.append([idxs[0]])
            cluster_of[idxs[0]] = cid
            continue
        Xc = rawX[idxs]
        Tc = termB[idxs]

        # Exact sparse equality avoids float32 roundoff at tau_term = 1.
        if term_sim == 'cosine' and tau_term is not None and float(tau_term) >= 1.0 - 1e-12:
            term_labels = exact_sparse_row_labels(Tc) if Tc.shape[1] > 0 else np.zeros(len(idxs), dtype=np.int32)
            term_groups = defaultdict(list)
            for pos, lab in enumerate(term_labels.tolist()):
                term_groups[int(lab)].append(pos)
        else:
            term_groups = {0: list(range(len(idxs)))}

        for local_group in term_groups.values():
            if len(local_group) == 1:
                cid = len(clusters)
                member = idxs[local_group[0]]
                clusters.append([member])
                cluster_of[member] = cid
                continue

            Xg = Xc[local_group]
            Tg = Tc[local_group]
            sim_attr, sim_term = layered_similarity_matrices(
                Xg,
                Tg,
                term_sim=term_sim if not (term_sim == 'cosine' and tau_term is not None and float(tau_term) >= 1.0 - 1e-12) else 'cosine'
            )
            if term_sim == 'cosine' and tau_term is not None and float(tau_term) >= 1.0 - 1e-12:
                sim_term = None
            unassigned = list(range(len(local_group)))
            while unassigned:
                seed = unassigned.pop(0)
                cluster = [seed]
                for cand in unassigned.copy():
                    if _passes_layered_guard(cand, cluster, sim_attr, sim_term, tau_attr=tau_attr, tau_term=tau_term):
                        cluster.append(cand)
                        unassigned.remove(cand)
                cid = len(clusters)
                members = [idxs[local_group[k]] for k in cluster]
                clusters.append(members)
                for m in members:
                    cluster_of[m] = cid
    return cluster_of, clusters


def make_threshold_mapping(parts_local, rawX, termB, tau, alpha=0.5):
    """Compatibility wrapper using one threshold for both channels."""
    return make_layered_threshold_mapping(parts_local, rawX, termB, tau_term=tau, tau_attr=tau, term_sim='cosine')


def build_group_matrix(cluster_of):
    n = len(cluster_of)
    n_clusters = int(cluster_of.max()) + 1 if n > 0 else 0
    rows = cluster_of
    cols = np.arange(n, dtype=np.int32)
    data = np.ones(n, dtype=np.float32)
    return sp.csr_matrix((data, (rows, cols)), shape=(n_clusters, n), dtype=np.float32)


def reduce_relation_matrix(M, st, et, G_by_type):
    out = M
    if st in G_by_type:
        out = G_by_type[st] @ out
    if et in G_by_type:
        out = out @ G_by_type[et].T
    return out.tocsr().astype(np.float32)


def reduced_path_matrix(mats, link_defs, link_seq, G_by_type):
    lt0 = link_seq[0]
    st0, et0, _ = link_defs[lt0]
    B = reduce_relation_matrix(mats[lt0], st0, et0, G_by_type)
    cur_type = et0
    for lt in link_seq[1:]:
        st, et, _ = link_defs[lt]
        assert st == cur_type
        B = (B @ reduce_relation_matrix(mats[lt], st, et, G_by_type)).tocsr().astype(np.float32)
        cur_type = et
    return B, cur_type


def pathsim_row_from_B(B, diag, i):
    prod = (B.getrow(i) @ B.T).toarray().ravel().astype(np.float32)
    denom = diag[i] + diag
    return np.divide(2.0 * prod, denom, out=np.zeros_like(prod), where=denom > 0)


def hetesim_row_from_path_pair(BL, BR, BL_norm, BR_norm, i):
    left = BL_norm.getrow(i)
    sim = (left @ BR_norm.T).toarray().ravel().astype(np.float32)
    return sim


def compute_hetesim_path_factors(mats, link_defs, link_seq, G_by_type):
    st0 = link_defs[link_seq[0]][0]
    B_all, end_t = reduced_path_matrix(mats, link_defs, link_seq, G_by_type)

    if st0 != end_t:
        BL = B_all
        BR = B_all
        BLn = normalize(BL, norm='l2', axis=1, copy=True)
        BRn = normalize(BR, norm='l2', axis=1, copy=True)
        return BL, BR, BLn.tocsr(), BRn.tocsr()

    if len(link_seq) % 2 != 0:
        raise ValueError('HeteSim full-path mode expects an even-length symmetric relation sequence.')
    half = len(link_seq) // 2
    left_seq = link_seq[:half]
    right_seq = link_seq[half:]
    BL, _ = reduced_path_matrix(mats, link_defs, left_seq, G_by_type)
    BR, _ = reduced_path_matrix(mats, link_defs, right_seq, G_by_type)
    BLn = normalize(BL, norm='l2', axis=1, copy=True)
    BRn = normalize(BR, norm='l2', axis=1, copy=True)
    return BL, BR, BLn.tocsr(), BRn.tocsr()


def deterministic_topk_indices(scores, k, node_ids=None):
    """Return top-k positions using score descending and node id ascending."""
    values = np.asarray(scores, dtype=np.float64)
    n = len(values)
    k = min(max(int(k), 0), n)
    if k == 0:
        return np.empty(0, dtype=np.int64)
    ids = np.arange(n, dtype=np.int64) if node_ids is None else np.asarray(node_ids)
    if len(ids) != n:
        raise ValueError('node_ids and scores must have the same length')
    if k == n:
        return np.lexsort((ids, -values))
    threshold = np.partition(values, n - k)[n - k]
    strict = np.flatnonzero(values > threshold)
    boundary = np.flatnonzero(values == threshold)
    slots = k - len(strict)
    boundary_order = np.argsort(ids[boundary], kind='stable')[:slots]
    selected = np.concatenate((strict, boundary[boundary_order]))
    return selected[np.lexsort((ids[selected], -values[selected]))]


def _tie_averaged_dcg_at_ks(order_scores, gains, k_values):
    scores = np.asarray(order_scores, dtype=np.float64)
    gains = np.asarray(gains, dtype=np.float64)
    max_k = min(max([max(k, 0) for k in k_values], default=0), len(scores))
    if max_k == 0:
        return {int(k): 0.0 for k in k_values}
    order = deterministic_topk_indices(scores, max_k)
    top_scores = scores[order]
    mean_gain = {
        score: float(np.mean(gains[scores == score]))
        for score in np.unique(top_scores)
    }
    expected_gains = np.asarray([mean_gain[score] for score in top_scores])
    discounts = 1.0 / np.log2(np.arange(2, max_k + 2))
    prefix = np.cumsum(expected_gains * discounts)
    return {
        int(k): (0.0 if min(max(int(k), 0), len(scores)) == 0 else float(prefix[min(int(k), len(scores)) - 1]))
        for k in k_values
    }


def ndcg_at_ks(pred_scores, true_scores, k_values):
    pred = np.asarray(pred_scores, dtype=np.float64)
    true = np.asarray(true_scores, dtype=np.float64)
    requested = [int(k) for k in k_values]
    dcg = _tie_averaged_dcg_at_ks(pred, true, requested)
    ideal = _tie_averaged_dcg_at_ks(true, true, requested)
    return {
        k: (dcg[k] / ideal[k] if ideal[k] > 1e-12 else 1.0)
        for k in requested
    }


def ndcg_at_k(pred_scores, true_scores, k=10):
    """Compute nDCG with expected gain within tied predicted-score groups."""
    return ndcg_at_ks(pred_scores, true_scores, [k])[int(k)]


def tie_aware_overlap_at_ks(scores_a, scores_b, k_values):
    """Fuzzy top-k overlaps using inclusion weights for boundary ties."""
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    requested = [int(k) for k in k_values]
    max_k = min(max([max(k, 0) for k in requested], default=0), len(a), len(b))
    order_a = deterministic_topk_indices(a, max_k)
    order_b = deterministic_topk_indices(b, max_k)

    def inclusion_weights(values, order, k):
        threshold = values[order[k - 1]]
        weights = np.zeros(len(values), dtype=np.float64)
        strict = values > threshold
        tied = values == threshold
        weights[strict] = 1.0
        slots = k - int(np.sum(strict))
        tie_count = int(np.sum(tied))
        if tie_count:
            weights[tied] = slots / tie_count
        return weights

    results = {}
    for requested_k in requested:
        k = min(max(requested_k, 0), len(a), len(b))
        if k == 0:
            results[requested_k] = 1.0
            continue
        weights_a = inclusion_weights(a, order_a, k)
        weights_b = inclusion_weights(b, order_b, k)
        results[requested_k] = float(np.minimum(weights_a, weights_b).sum() / k)
    return results


def tie_aware_overlap_at_k(scores_a, scores_b, k):
    return tie_aware_overlap_at_ks(scores_a, scores_b, [k])[int(k)]


def safe_paired_ttest_pvalue(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b):
        n = min(len(a), len(b))
        a = a[:n]
        b = b[:n]
    if len(a) < 2:
        return 1.0
    if np.allclose(a, b, atol=1e-12, rtol=1e-12):
        return 1.0
    try:
        res = stats.ttest_rel(a, b, nan_policy='omit')
        p = float(res.pvalue)
        if np.isnan(p):
            return 1.0
        return p
    except Exception:
        return 1.0


def ranking_metrics(sim_o, sim_r, self_idx, k_values=(1, 5, 10)):
    so = sim_o.copy()
    sr = sim_r.copy()
    so[self_idx] = -1.0
    sr[self_idx] = -1.0
    abs_diff = np.abs(so - sr)
    # Percentile errors are more stable than MaxE on large graphs because MaxE
    # can be dominated by a few extreme node pairs.  We still report MaxE as a
    # diagnostic indicator, but P95E / P99E are recommended for constraints.
    out = {
        'mae': float(np.mean(abs_diff)),
        'p95e': float(np.percentile(abs_diff, 95)),
        'p99e': float(np.percentile(abs_diff, 99)),
        'maxe': float(np.max(abs_diff)),
        'pvalue': safe_paired_ttest_pvalue(so, sr),
    }
    max_k = min(max(k_values, default=0), len(so) - 1)
    order_o = deterministic_topk_indices(so, max_k)
    order_r = deterministic_topk_indices(sr, max_k)
    for k in k_values:
        kk = min(k, len(so) - 1)
        if kk <= 0:
            out[f'top{k}'] = 1.0
            continue
        top_o = order_o[:kk]
        top_r = order_r[:kk]
        out[f'top{k}'] = len(set(top_o.tolist()) & set(top_r.tolist())) / kk
    out['ndcg10'] = ndcg_at_k(sr, so, k=10)
    return out


def aggregate_row_metrics(metric_rows: List[dict], prefix: str) -> dict:
    out = {}
    # Average the per-node errors; retain MaxE as a worst-case diagnostic.
    for metric in ['mae', 'p95e', 'p99e', 'top1', 'top5', 'top10', 'ndcg10', 'pvalue']:
        out[f'{prefix}_{metric}'] = float(np.mean([r[f'{prefix}_{metric}'] for r in metric_rows]))
    # MaxE remains a worst-case diagnostic; aggregate it conservatively.
    out[f'{prefix}_maxe'] = float(np.max([r[f'{prefix}_maxe'] for r in metric_rows]))
    return out


def build_original_semantic_cache(data, mats, cfg):
    """Precompute original-path factors once per dataset.

    Earlier versions recomputed original PathSim/HeteSim factors for every
    threshold pair. Since original graph semantics do not depend on tau, caching
    them significantly reduces repeated work.
    """
    cache = {}
    for pname, link_seq in cfg['meta_paths'].items():
        B_orig = compose_path(mats, link_seq)
        diag_orig = np.asarray(B_orig.multiply(B_orig).sum(axis=1)).ravel().astype(np.float32)
        BL_orig, BR_orig, BLn_orig, BRn_orig = compute_hetesim_path_factors(mats, data['link_defs'], link_seq, {})
        cache[pname] = {
            'link_seq': link_seq,
            'B_orig': B_orig,
            'diag_orig': diag_orig,
            'BL_orig': BL_orig,
            'BR_orig': BR_orig,
            'BLn_orig': BLn_orig,
            'BRn_orig': BRn_orig,
        }
    return cache


def evaluate_semantics(data, mats, cfg, cluster_of_by_type, sample_idx=None, orig_cache=None):
    target_t = cfg['target_type']
    G_by_type = {t: build_group_matrix(cluster_of_by_type[t]) for t in cluster_of_by_type}
    target_cluster_of = cluster_of_by_type[target_t]
    n_orig = len(target_cluster_of)
    if sample_idx is None:
        sample_idx = np.arange(n_orig)
    path_rows = []

    for pname, link_seq in cfg['meta_paths'].items():
        if orig_cache is not None and pname in orig_cache:
            oc = orig_cache[pname]
            B_orig = oc['B_orig']
            diag_orig = oc['diag_orig']
            BL_orig = oc['BL_orig']
            BR_orig = oc['BR_orig']
            BLn_orig = oc['BLn_orig']
            BRn_orig = oc['BRn_orig']
        else:
            B_orig = compose_path(mats, link_seq)
            diag_orig = np.asarray(B_orig.multiply(B_orig).sum(axis=1)).ravel().astype(np.float32)
            BL_orig, BR_orig, BLn_orig, BRn_orig = compute_hetesim_path_factors(mats, data['link_defs'], link_seq, {})

        B_red, _ = reduced_path_matrix(mats, data['link_defs'], link_seq, G_by_type)
        diag_red = np.asarray(B_red.multiply(B_red).sum(axis=1)).ravel().astype(np.float32)
        BL_red, BR_red, BLn_red, BRn_red = compute_hetesim_path_factors(mats, data['link_defs'], link_seq, G_by_type)

        row_metrics = []
        for i in sample_idx:
            ci = target_cluster_of[i]
            sim_o = pathsim_row_from_B(B_orig, diag_orig, i)
            sim_r_red = pathsim_row_from_B(B_red, diag_red, ci)
            sim_r = sim_r_red[target_cluster_of]
            m = ranking_metrics(sim_o, sim_r, i)
            m = {f'pathsim_{k}': v for k, v in m.items()}

            hs_o = hetesim_row_from_path_pair(BL_orig, BR_orig, BLn_orig, BRn_orig, i)
            hs_r_red = hetesim_row_from_path_pair(BL_red, BR_red, BLn_red, BRn_red, ci)
            hs_r = hs_r_red[target_cluster_of]
            m2 = ranking_metrics(hs_o, hs_r, i)
            m2 = {f'hetesim_{k}': v for k, v in m2.items()}

            row_metrics.append({**m, **m2})

        row = {'path': pname, 'sample_size': int(len(sample_idx))}
        row.update(aggregate_row_metrics(row_metrics, 'pathsim'))
        row.update(aggregate_row_metrics(row_metrics, 'hetesim'))
        path_rows.append(row)

    agg = {}
    for metric in ['mae', 'p95e', 'p99e', 'top1', 'top5', 'top10', 'ndcg10', 'pvalue']:
        agg[f'pathsim_{metric}'] = float(np.mean([r[f'pathsim_{metric}'] for r in path_rows]))
        agg[f'hetesim_{metric}'] = float(np.mean([r[f'hetesim_{metric}'] for r in path_rows]))
    agg['pathsim_maxe'] = float(np.max([r['pathsim_maxe'] for r in path_rows]))
    agg['hetesim_maxe'] = float(np.max([r['hetesim_maxe'] for r in path_rows]))
    return path_rows, agg


def count_original_edges(data) -> int:
    return int(sum(len(v) for v in data['edges_by_link'].values()))


def count_reduced_edges(data, mats, cluster_of_by_type) -> int:
    G_by_type = {t: build_group_matrix(cluster_of_by_type[t]) for t in cluster_of_by_type}
    total = 0
    for lt, M in mats.items():
        st, et, _ = data['link_defs'][lt]
        M_red = reduce_relation_matrix(M, st, et, G_by_type)
        total += int(M_red.nnz)
    return total


def pareto_frontier(df: pd.DataFrame, reduction_col: str, quality_col: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    rows = []
    for idx, row in df.iterrows():
        dominated = False
        for jdx, other in df.iterrows():
            if idx == jdx:
                continue
            better_or_equal = other[reduction_col] >= row[reduction_col] and other[quality_col] >= row[quality_col]
            strictly_better = other[reduction_col] > row[reduction_col] or other[quality_col] > row[quality_col]
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            rows.append(row.to_dict())
    return pd.DataFrame(rows)


def plot_dataset(df: pd.DataFrame, dataset: str, outdir: str):
    sub = df[(df['dataset'] == dataset) & (df['mode'] == 'approx')].copy()
    if sub.empty:
        return
    sub['term_threshold_num'] = pd.to_numeric(sub['term_threshold'], errors='coerce')
    sub['attr_threshold_num'] = pd.to_numeric(sub['attr_threshold'], errors='coerce')
    sub = sub.dropna(subset=['term_threshold_num', 'attr_threshold_num'])
    if sub.empty:
        return

    fig = plt.figure(figsize=(8.4, 5.6))
    sc = plt.scatter(
        sub['fullgraph_reduction_ratio'],
        sub['semantic_score'],
        c=sub['term_threshold_num'],
        s=45 + 120 * sub['attr_threshold_num'],
        alpha=0.85,
    )
    plt.colorbar(sc, label='Term threshold')
    plt.xlabel('Full-graph node reduction ratio')
    plt.ylabel('Semantic score')
    plt.title(f'{dataset}: layered Guard reduction vs semantic quality')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'{dataset.lower()}_layered_tradeoff_scatter.png'), dpi=220)
    plt.close(fig)

    # Heatmap-style pivot for controllability, averaged if duplicate rows exist.
    try:
        pivot = sub.pivot_table(index='attr_threshold_num', columns='term_threshold_num', values='semantic_score', aggfunc='mean')
        fig = plt.figure(figsize=(8.2, 5.8))
        plt.imshow(pivot.values, aspect='auto', origin='lower')
        plt.xticks(np.arange(len(pivot.columns)), [f'{x:.2g}' for x in pivot.columns], rotation=45)
        plt.yticks(np.arange(len(pivot.index)), [f'{x:.2g}' for x in pivot.index])
        plt.xlabel('Term threshold')
        plt.ylabel('Attribute threshold')
        plt.title(f'{dataset}: semantic score heatmap')
        plt.colorbar(label='Semantic score')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'{dataset.lower()}_layered_semantic_score_heatmap.png'), dpi=220)
        plt.close(fig)
    except Exception:
        pass

def _float_grid(explicit: Optional[str], vmin: float, vmax: float, step: float) -> List[float]:
    if explicit:
        return [float(x) for x in explicit.split(',') if x.strip()]
    vals = []
    cur = vmin
    while cur <= vmax + 1e-12:
        vals.append(round(cur, 6))
        cur += step
    return vals


def threshold_pairs_from_args(args) -> List[Tuple[float, float]]:
    # Backward compatibility: --taus uses the same grid for both layers.
    if args.taus and not args.term_taus and not args.attr_taus:
        base = _float_grid(args.taus, args.threshold_min, args.threshold_max, args.threshold_step)
        return [(t, t) for t in base]
    term_vals = _float_grid(args.term_taus, args.term_threshold_min, args.term_threshold_max, args.term_threshold_step)
    attr_vals = _float_grid(args.attr_taus, args.attr_threshold_min, args.attr_threshold_max, args.attr_threshold_step)
    return [(tt, ta) for tt in term_vals for ta in attr_vals]


def semantic_quality_scores(agg: dict) -> dict:
    """Separate quality scores for fallback recommendation.

    These scores are not the formal constraints.  Formal constraints are the
    boolean flags computed by controllability_flags().  P95E and P99E replace
    MaxE in the main score because they are more robust on real graphs.
    """
    pathsim_score = (
        agg['pathsim_ndcg10'] + agg['pathsim_top10'] + agg['pathsim_pvalue']
        - agg['pathsim_mae'] - 0.50 * agg['pathsim_p95e'] - 0.25 * agg['pathsim_p99e']
    )
    hetesim_score = (
        agg['hetesim_ndcg10'] + agg['hetesim_top10'] + agg['hetesim_pvalue']
        - agg['hetesim_mae'] - 0.50 * agg['hetesim_p95e'] - 0.25 * agg['hetesim_p99e']
    )
    return {
        'pathsim_score': float(pathsim_score),
        'hetesim_score': float(hetesim_score),
        'joint_score': float(pathsim_score + hetesim_score),
    }


def controllability_flags(agg: dict, args) -> dict:
    """Compute separate and joint semantic controllability flags.

    PathSim and HeteSim are constrained separately.  P95E / P99E are used as
    robust local-error bounds.  MaxE can optionally be used as an additional
    worst-case diagnostic constraint by passing --use-maxe-constraint.
    """
    pathsim_ok = (
        agg['pathsim_mae'] <= args.max_pathsim_mae and
        agg['pathsim_p95e'] <= args.max_pathsim_p95e and
        agg['pathsim_p99e'] <= args.max_pathsim_p99e and
        agg['pathsim_top1'] >= args.min_pathsim_top1 and
        agg['pathsim_top5'] >= args.min_pathsim_top5 and
        agg['pathsim_top10'] >= args.min_pathsim_top10 and
        agg['pathsim_ndcg10'] >= args.min_pathsim_ndcg10 and
        agg['pathsim_pvalue'] >= args.min_pathsim_pvalue
    )
    hetesim_ok = (
        agg['hetesim_mae'] <= args.max_hetesim_mae and
        agg['hetesim_p95e'] <= args.max_hetesim_p95e and
        agg['hetesim_p99e'] <= args.max_hetesim_p99e and
        agg['hetesim_top1'] >= args.min_hetesim_top1 and
        agg['hetesim_top5'] >= args.min_hetesim_top5 and
        agg['hetesim_top10'] >= args.min_hetesim_top10 and
        agg['hetesim_ndcg10'] >= args.min_hetesim_ndcg10 and
        agg['hetesim_pvalue'] >= args.min_hetesim_pvalue
    )
    if getattr(args, 'use_maxe_constraint', False):
        pathsim_ok = pathsim_ok and (agg['pathsim_maxe'] <= args.max_pathsim_maxe)
        hetesim_ok = hetesim_ok and (agg['hetesim_maxe'] <= args.max_hetesim_maxe)
    pathsim_controllable = int(pathsim_ok)
    hetesim_controllable = int(hetesim_ok)
    return {
        'pathsim_controllable': pathsim_controllable,
        'hetesim_controllable': hetesim_controllable,
        'joint_controllable': int(pathsim_controllable and hetesim_controllable),
        'controllable': int(pathsim_controllable and hetesim_controllable),
    }


def choose_recommendation(df: pd.DataFrame, criterion: str) -> dict:
    """Choose the best threshold pair for pathsim/hetesim/joint.

    Primary rule: among feasible threshold pairs, maximize reduction ratio.
    Fallback rule: if no feasible pair exists, maximize the corresponding quality
    score and then the reduction ratio.
    """
    if df.empty:
        return {}
    if criterion == 'pathsim':
        flag = 'pathsim_controllable'
        score = 'pathsim_score'
        sort_good = ['fullgraph_reduction_ratio', 'fullgraph_edge_reduction_ratio',
                     'avg_pathsim_ndcg10', 'avg_pathsim_top10', 'avg_pathsim_top5',
                     'avg_pathsim_pvalue']
        sort_fallback = [score, 'fullgraph_reduction_ratio', 'avg_pathsim_ndcg10', 'avg_pathsim_top10']
    elif criterion == 'hetesim':
        flag = 'hetesim_controllable'
        score = 'hetesim_score'
        sort_good = ['fullgraph_reduction_ratio', 'fullgraph_edge_reduction_ratio',
                     'avg_hetesim_ndcg10', 'avg_hetesim_top10', 'avg_hetesim_top5',
                     'avg_hetesim_pvalue']
        sort_fallback = [score, 'fullgraph_reduction_ratio', 'avg_hetesim_ndcg10', 'avg_hetesim_top10']
    elif criterion == 'joint':
        flag = 'joint_controllable'
        score = 'joint_score'
        sort_good = ['fullgraph_reduction_ratio', 'fullgraph_edge_reduction_ratio',
                     'avg_pathsim_ndcg10', 'avg_hetesim_ndcg10',
                     'avg_pathsim_top10', 'avg_hetesim_top10']
        sort_fallback = [score, 'fullgraph_reduction_ratio', 'avg_pathsim_ndcg10', 'avg_hetesim_ndcg10']
    else:
        raise ValueError(f'Unknown recommendation criterion: {criterion}')

    good = df[df[flag].astype(int) == 1].copy() if flag in df.columns else pd.DataFrame()
    if not good.empty:
        sort_cols = [c for c in sort_good if c in good.columns]
        chosen = good.sort_values(sort_cols, ascending=[False] * len(sort_cols)).iloc[0]
        out = chosen.to_dict()
        out['recommendation_criterion'] = criterion
        out['recommendation_status'] = 'feasible'
        return out

    sort_cols = [c for c in sort_fallback if c in df.columns]
    chosen = df.sort_values(sort_cols, ascending=[False] * len(sort_cols)).iloc[0] if sort_cols else df.iloc[0]
    out = chosen.to_dict()
    out['recommendation_criterion'] = criterion
    out['recommendation_status'] = 'fallback_no_feasible_threshold'
    return out



_WORKER_STATE = None


def _strip_unpickleable_data(data: dict) -> dict:
    """Remove fields that are not needed in workers and may not pickle cleanly."""
    out = dict(data)
    out.pop('zip', None)
    return out


def _init_threshold_worker(state: dict):
    global _WORKER_STATE
    _WORKER_STATE = state


def _evaluate_threshold_pair_worker(pair):
    return evaluate_threshold_pair(pair, _WORKER_STATE)


def evaluate_threshold_pair(pair, state: dict) -> dict:
    """Evaluate one threshold pair for one dataset.

    The function is top-level so it can be used by ProcessPoolExecutor on
    Windows. It returns all rows needed by the main process.
    """
    threshold_t0 = time.perf_counter()
    name = state['name']
    cfg = state['cfg']
    data = state['data']
    mats = state['mats']
    parts = state['parts']
    term_sim = state.get('term_sim', 'cosine')
    rawX_by_type = state['rawX_by_type']
    termB_by_type = state['termB_by_type']
    type_names = state['type_names']
    sample_idx = state['sample_idx']
    full_nodes = state['full_nodes']
    main_nodes = state['main_nodes']
    semantic_nodes = state['semantic_nodes']
    original_full_edges = state['original_full_edges']
    orig_cache = state.get('orig_cache')
    args = state['args']

    mode = 'exact-baseline' if pair is None else 'approx'
    tau_term = None if pair is None else float(pair[0])
    tau_attr = None if pair is None else float(pair[1])
    threshold_label = None if pair is None else f'T{tau_term:.6g}_A{tau_attr:.6g}'
    cluster_of_by_type = {}
    reduced_counts = {}
    pertype_rows = []

    for t in cfg['main_types']:
        if pair is None:
            cluster_of = np.arange(len(parts[t]), dtype=np.int32)
            clusters = [[i] for i in range(len(parts[t]))]
        else:
            cluster_of, clusters = make_layered_threshold_mapping(
                parts[t], rawX_by_type[t], termB_by_type[t], tau_term=tau_term, tau_attr=tau_attr, term_sim=term_sim
            )
        cluster_of_by_type[t] = cluster_of
        reduced_counts[t] = len(clusters)
        pertype_rows.append({
            'dataset': name,
            'mode': mode,
            'threshold': threshold_label,
            'term_threshold': tau_term,
            'attr_threshold': tau_attr,
            'term_sim': term_sim,
            'type_id': t,
            'type_name': type_names[t],
            'original_nodes': len(cluster_of),
            'reduced_nodes': len(clusters),
            'reduction_ratio': 1.0 - len(clusters) / len(cluster_of),
            'strong_attr_enabled': int(t in cfg['strong_attr_types']),
        })

    for t in cfg['semantic_types']:
        n = len(data['ids_by_type'][t])
        cluster_of_by_type[t] = np.arange(n, dtype=np.int32)
        reduced_counts[t] = n
        pertype_rows.append({
            'dataset': name,
            'mode': mode,
            'threshold': threshold_label,
            'term_threshold': tau_term,
            'attr_threshold': tau_attr,
            'term_sim': term_sim,
            'type_id': t,
            'type_name': type_names[t],
            'original_nodes': n,
            'reduced_nodes': n,
            'reduction_ratio': 0.0,
            'strong_attr_enabled': 0,
        })

    path_rows, agg = evaluate_semantics(data, mats, cfg, cluster_of_by_type, sample_idx=sample_idx, orig_cache=orig_cache)
    full_reduced_nodes = int(sum(reduced_counts[t] for t in cfg['main_types']) + semantic_nodes)
    full_reduction_ratio = 1.0 - full_reduced_nodes / full_nodes
    main_reduced_nodes = int(sum(reduced_counts[t] for t in cfg['main_types']))
    main_reduction_ratio = 1.0 - main_reduced_nodes / main_nodes
    reduced_full_edges = count_reduced_edges(data, mats, cluster_of_by_type)
    full_edge_reduction_ratio = 1.0 - reduced_full_edges / original_full_edges if original_full_edges > 0 else 0.0
    scores = semantic_quality_scores(agg)
    flags = controllability_flags(agg, args)
    semantic_score = scores['joint_score']
    threshold_runtime_sec = time.perf_counter() - threshold_t0
    try:
        current_peak_mb = tracemalloc.get_traced_memory()[1] / (1024 * 1024) if tracemalloc.is_tracing() else np.nan
    except Exception:
        current_peak_mb = np.nan

    result_row = {
        'dataset': name,
        'mode': mode,
        'threshold': threshold_label,
        'term_threshold': tau_term,
        'attr_threshold': tau_attr,
        'term_sim': term_sim,
        'target_type': type_names[cfg['target_type']],
        'strong_attr_types': '+'.join(type_names[t] for t in cfg['strong_attr_types']),
        'main_structure_types': '+'.join(type_names[t] for t in cfg['main_types']),
        'semantic_types': '+'.join(type_names[t] for t in cfg['semantic_types']),
        'target_sample_size': int(len(sample_idx)),
        'original_fullgraph_nodes': full_nodes,
        'reduced_fullgraph_nodes': full_reduced_nodes,
        'fullgraph_reduction_ratio': float(full_reduction_ratio),
        'original_main_nodes': main_nodes,
        'reduced_main_nodes': main_reduced_nodes,
        'main_reduction_ratio': float(main_reduction_ratio),
        'original_fullgraph_edges': int(original_full_edges),
        'reduced_fullgraph_edges': int(reduced_full_edges),
        'fullgraph_edge_reduction_ratio': float(full_edge_reduction_ratio),
        'avg_pathsim_mae': agg['pathsim_mae'],
        'avg_pathsim_p95e': agg['pathsim_p95e'],
        'avg_pathsim_p99e': agg['pathsim_p99e'],
        'avg_pathsim_maxe': agg['pathsim_maxe'],
        'avg_pathsim_top1': agg['pathsim_top1'],
        'avg_pathsim_top5': agg['pathsim_top5'],
        'avg_pathsim_top10': agg['pathsim_top10'],
        'avg_pathsim_ndcg10': agg['pathsim_ndcg10'],
        'avg_pathsim_pvalue': agg['pathsim_pvalue'],
        'avg_hetesim_mae': agg['hetesim_mae'],
        'avg_hetesim_p95e': agg['hetesim_p95e'],
        'avg_hetesim_p99e': agg['hetesim_p99e'],
        'avg_hetesim_maxe': agg['hetesim_maxe'],
        'avg_hetesim_top1': agg['hetesim_top1'],
        'avg_hetesim_top5': agg['hetesim_top5'],
        'avg_hetesim_top10': agg['hetesim_top10'],
        'avg_hetesim_ndcg10': agg['hetesim_ndcg10'],
        'avg_hetesim_pvalue': agg['hetesim_pvalue'],
        'pathsim_score': scores['pathsim_score'],
        'hetesim_score': scores['hetesim_score'],
        'semantic_score': float(semantic_score),
        'pathsim_controllable': flags['pathsim_controllable'],
        'hetesim_controllable': flags['hetesim_controllable'],
        'joint_controllable': flags['joint_controllable'],
        'threshold_runtime_sec': float(threshold_runtime_sec),
        'python_peak_mem_mb': float(current_peak_mb) if not pd.isna(current_peak_mb) else np.nan,
        'controllable': flags['joint_controllable'],
    }

    perpath_rows = []
    for row in path_rows:
        row2 = {
            'dataset': name,
            'mode': mode,
            'threshold': threshold_label,
            'term_threshold': tau_term,
            'attr_threshold': tau_attr,
            'term_sim': term_sim,
            'semantic_score': float(semantic_score),
        }
        row2.update(row)
        perpath_rows.append(row2)

    return {
        'result': result_row,
        'pertype': pertype_rows,
        'perpath': perpath_rows,
    }


def main():
    ap = argparse.ArgumentParser(description='HHIN semantic-preservation benchmark with layered Guards, P95E/P99E constraints, and optional CPU parallelism')
    ap.add_argument('--base-dir', type=str, default='.')
    ap.add_argument('--out-dir', type=str, default='./results/reduction_analysis')
    ap.add_argument('--datasets', nargs='*', default=['ACM', 'DBLP', 'IMDB'])
    ap.add_argument('--alpha', type=float, default=0.5, help='Deprecated: kept for compatibility; layered Guard uses separate term/attribute thresholds.')
    ap.add_argument('--sample-size', type=int, default=0, help='0 means all target nodes. Use a positive number for faster exploratory runs.')
    ap.add_argument('--n-jobs', type=int, default=1, help='Number of CPU worker processes per dataset. Use 1 for serial execution.')
    ap.add_argument('--taus', type=str, default=None, help='Backward-compatible comma-separated thresholds; uses the same values for term and attribute if --term-taus/--attr-taus are absent.')
    ap.add_argument('--threshold-min', type=float, default=0.1)
    ap.add_argument('--threshold-max', type=float, default=1.0)
    ap.add_argument('--threshold-step', type=float, default=0.1, help='Backward-compatible step when --taus is used for both layers')
    ap.add_argument('--term-taus', type=str, default=None, help='comma-separated thresholds for semantic-channel/term similarity')
    ap.add_argument('--term-threshold-min', type=float, default=0.1)
    ap.add_argument('--term-threshold-max', type=float, default=1.0)
    ap.add_argument('--term-threshold-step', type=float, default=0.1)
    ap.add_argument('--attr-taus', type=str, default=None, help='comma-separated thresholds for node-attached attribute similarity')
    ap.add_argument('--attr-threshold-min', type=float, default=0.1)
    ap.add_argument('--attr-threshold-max', type=float, default=1.0)
    ap.add_argument('--attr-threshold-step', type=float, default=0.1)
    ap.add_argument('--term-sim', type=str, default='cosine', choices=['cosine', 'jaccard', 'overlap'],
                    help='Similarity for term/keyword semantic-channel Guard. Default cosine is smoother; jaccard is stricter ablation; overlap is permissive containment ablation.')

    # Robust semantic-error constraints. P95E/P99E are preferred over MaxE as hard constraints.
    ap.add_argument('--max-pathsim-mae', type=float, default=0.10)
    ap.add_argument('--max-pathsim-p95e', type=float, default=0.30)
    ap.add_argument('--max-pathsim-p99e', type=float, default=0.50)
    ap.add_argument('--max-pathsim-maxe', type=float, default=1.0, help='Used only if --use-maxe-constraint is set; otherwise reported as diagnostic.')
    ap.add_argument('--min-pathsim-top1', type=float, default=0.0)
    ap.add_argument('--min-pathsim-top5', type=float, default=0.0)
    ap.add_argument('--min-pathsim-top10', type=float, default=0.80)
    ap.add_argument('--min-pathsim-ndcg10', type=float, default=0.95)
    ap.add_argument('--min-pathsim-pvalue', type=float, default=0.05)
    ap.add_argument('--max-hetesim-mae', type=float, default=0.10)
    ap.add_argument('--max-hetesim-p95e', type=float, default=0.30)
    ap.add_argument('--max-hetesim-p99e', type=float, default=0.50)
    ap.add_argument('--max-hetesim-maxe', type=float, default=1.0, help='Used only if --use-maxe-constraint is set; otherwise reported as diagnostic.')
    ap.add_argument('--min-hetesim-top1', type=float, default=0.0)
    ap.add_argument('--min-hetesim-top5', type=float, default=0.0)
    ap.add_argument('--min-hetesim-top10', type=float, default=0.80)
    ap.add_argument('--min-hetesim-ndcg10', type=float, default=0.95)
    ap.add_argument('--min-hetesim-pvalue', type=float, default=0.05)
    ap.add_argument('--use-maxe-constraint', action='store_true', help='Also require MaxE thresholds. Not recommended for large graphs unless deliberately strict.')
    ap.add_argument('--write-pareto', action='store_true', help='write dataset-wise Pareto frontier CSV')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tracemalloc.start()
    global_t0 = time.perf_counter()

    results = []
    perpath = []
    pertype = []
    partsum = []
    pathsim_recommendations = []
    hetesim_recommendations = []
    joint_recommendations = []
    pareto_rows = []

    threshold_pairs = threshold_pairs_from_args(args)

    for name in args.datasets:
        dataset_t0 = time.perf_counter()
        cfg = CONFIGS[name]
        data_loaded = load_hgb_zip(os.path.join(args.base_dir, f'{name}.zip'))
        mats = build_link_matrices(data_loaded)
        type_names = TYPE_NAME_CACHE[data_loaded['path']]
        data = _strip_unpickleable_data(data_loaded)
        parts, history = fixed_point_partition(data, mats, cfg['core_link_types'])

        for t in cfg['main_types'] + cfg['semantic_types']:
            p = parts[t]
            uniq, cnts = np.unique(p, return_counts=True)
            partsum.append({
                'dataset': name,
                'type_id': t,
                'type_name': type_names[t],
                'nodes': len(p),
                'struct_candidate_classes': len(uniq),
                'struct_non_singleton_classes': int(np.sum(cnts > 1)),
                'struct_max_class_size': int(cnts.max()),
                'struct_reduction_upper_bound': 1.0 - len(uniq) / len(p),
                'fixed_point_iters': len(history),
            })

        rawX_by_type = {}
        termB_by_type = {}
        for t in cfg['main_types']:
            rawX_by_type[t] = get_raw_attr_for_type(data, t, cfg['strong_attr_types'])
            termB_by_type[t] = derive_term_matrix(mats, cfg['term_paths_by_type'][t]).tocsr().astype(np.float32)

        target_n = len(data['ids_by_type'][cfg['target_type']])
        if args.sample_size and args.sample_size > 0:
            rng = np.random.default_rng(20260410)
            sample_idx = np.sort(rng.choice(target_n, size=min(args.sample_size, target_n), replace=False))
        else:
            sample_idx = np.arange(target_n)

        full_nodes = int(data['N'])
        main_nodes = int(sum(len(data['ids_by_type'][t]) for t in cfg['main_types']))
        semantic_nodes = int(sum(len(data['ids_by_type'][t]) for t in cfg['semantic_types']))
        original_full_edges = count_original_edges(data)
        orig_cache = build_original_semantic_cache(data, mats, cfg)

        state = {
            'name': name,
            'cfg': cfg,
            'data': data,
            'mats': mats,
            'parts': parts,
            'rawX_by_type': rawX_by_type,
            'termB_by_type': termB_by_type,
            'type_names': type_names,
            'sample_idx': sample_idx,
            'full_nodes': full_nodes,
            'main_nodes': main_nodes,
            'semantic_nodes': semantic_nodes,
            'original_full_edges': original_full_edges,
            'orig_cache': orig_cache,
            'args': args,
            'term_sim': args.term_sim,
        }
        tasks = [None] + threshold_pairs

        if args.n_jobs and args.n_jobs > 1 and len(tasks) > 1:
            # Threshold-pair CPU parallelism.  SciPy sparse kernels used here are CPU-bound.
            # Multi-GPU acceleration would require a CuPy/PyTorch sparse backend and is not used here.
            with ProcessPoolExecutor(max_workers=int(args.n_jobs), initializer=_init_threshold_worker, initargs=(state,)) as ex:
                futures = [ex.submit(_evaluate_threshold_pair_worker, pair) for pair in tasks]
                for fut in as_completed(futures):
                    pack = fut.result()
                    results.append(pack['result'])
                    pertype.extend(pack['pertype'])
                    perpath.extend(pack['perpath'])
        else:
            for pair in tasks:
                pack = evaluate_threshold_pair(pair, state)
                results.append(pack['result'])
                pertype.extend(pack['pertype'])
                perpath.extend(pack['perpath'])

        dataset_df = pd.DataFrame([r for r in results if r['dataset'] == name and r['mode'] == 'approx'])
        if not dataset_df.empty:
            dataset_df = dataset_df.sort_values(['term_threshold', 'attr_threshold'], na_position='first')
            for criterion, bucket in [
                ('pathsim', pathsim_recommendations),
                ('hetesim', hetesim_recommendations),
                ('joint', joint_recommendations),
            ]:
                chosen_dict = choose_recommendation(dataset_df, criterion)
                if chosen_dict:
                    chosen_dict['dataset_runtime_sec'] = float(time.perf_counter() - dataset_t0)
                    bucket.append(chosen_dict)

            frontier = pareto_frontier(dataset_df, 'fullgraph_reduction_ratio', 'semantic_score')
            if not frontier.empty:
                frontier = frontier.copy()
                frontier['dataset'] = name
                pareto_rows.extend(frontier.to_dict('records'))

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values(['dataset', 'mode', 'term_threshold', 'attr_threshold'], na_position='first')
    perpath_df = pd.DataFrame(perpath)
    if not perpath_df.empty:
        perpath_df = perpath_df.sort_values(['dataset', 'mode', 'path', 'term_threshold', 'attr_threshold'], na_position='first')
    pertype_df = pd.DataFrame(pertype)
    if not pertype_df.empty:
        pertype_df = pertype_df.sort_values(['dataset', 'mode', 'type_id', 'term_threshold', 'attr_threshold'], na_position='first')
    part_df = pd.DataFrame(partsum)
    pathsim_rec_df = pd.DataFrame(pathsim_recommendations)
    hetesim_rec_df = pd.DataFrame(hetesim_recommendations)
    joint_rec_df = pd.DataFrame(joint_recommendations)
    pareto_df = pd.DataFrame(pareto_rows)

    results_df.to_csv(os.path.join(args.out_dir, 'semantic_preservation_summary.csv'), index=False)
    perpath_df.to_csv(os.path.join(args.out_dir, 'semantic_preservation_per_path.csv'), index=False)
    pertype_df.to_csv(os.path.join(args.out_dir, 'semantic_preservation_per_type.csv'), index=False)
    part_df.to_csv(os.path.join(args.out_dir, 'semantic_preservation_struct_partition_summary.csv'), index=False)
    joint_rec_df.to_csv(os.path.join(args.out_dir, 'semantic_preservation_recommended_thresholds.csv'), index=False)
    pathsim_rec_df.to_csv(os.path.join(args.out_dir, 'pathsim_recommended_thresholds.csv'), index=False)
    hetesim_rec_df.to_csv(os.path.join(args.out_dir, 'hetesim_recommended_thresholds.csv'), index=False)
    joint_rec_df.to_csv(os.path.join(args.out_dir, 'joint_recommended_thresholds.csv'), index=False)
    if args.write_pareto:
        pareto_df.to_csv(os.path.join(args.out_dir, 'semantic_preservation_pareto_frontier.csv'), index=False)

    for ds in args.datasets:
        plot_dataset(results_df, ds, args.out_dir)

    peak_mem_mb = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
    total_runtime_sec = time.perf_counter() - global_t0
    tracemalloc.stop()

    with open(os.path.join(args.out_dir, 'run_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('HHIN approximate-reduction benchmark with layered Guards, P95E/P99E, and CPU parallelism\n')
        f.write('Stage 1: main-structure fixed-point refinement with equivalence-class identifiers\n')
        f.write('Stage 2: layered Guard approximate reduction on main-structure candidate classes only\n')
        f.write('Guard rule: term similarity >= tau_term AND node-attached attribute similarity >= tau_attr\n')
        f.write(f'Term similarity mode: {args.term_sim}\n')
        f.write('Semantic-channel nodes are preserved as pass-through singletons in the full graph\n')
        f.write('Primary evaluation: PathSim/HeteSim similarity and ranking preservation\n')
        f.write('Metrics: node/edge reduction, MAE, P95E, P99E, MaxE diagnostic, top-k consistency, nDCG@10, paired t-test p-values\n')
        f.write(
            f'PathSim constraints: MAE<={args.max_pathsim_mae}, P95E<={args.max_pathsim_p95e}, '
            f'P99E<={args.max_pathsim_p99e}, '
            f'MaxE<={args.max_pathsim_maxe} if use_maxe_constraint={args.use_maxe_constraint}, '
            f'Top1>={args.min_pathsim_top1}, Top5>={args.min_pathsim_top5}, '
            f'Top10>={args.min_pathsim_top10}, nDCG@10>={args.min_pathsim_ndcg10}, '
            f'p>={args.min_pathsim_pvalue}\n'
        )
        f.write(
            f'HeteSim constraints: MAE<={args.max_hetesim_mae}, P95E<={args.max_hetesim_p95e}, '
            f'P99E<={args.max_hetesim_p99e}, '
            f'MaxE<={args.max_hetesim_maxe} if use_maxe_constraint={args.use_maxe_constraint}, '
            f'Top1>={args.min_hetesim_top1}, Top5>={args.min_hetesim_top5}, '
            f'Top10>={args.min_hetesim_top10}, nDCG@10>={args.min_hetesim_ndcg10}, '
            f'p>={args.min_hetesim_pvalue}\n'
        )
        f.write('Recommended thresholds are exported separately for PathSim, HeteSim, and joint-controllable settings.\n')
        f.write(f'Layered threshold pairs (term, attribute): {threshold_pairs}\n')
        f.write(f'n_jobs: {args.n_jobs}\n')
        f.write(f'Total runtime (sec): {total_runtime_sec:.4f}\n')
        f.write(f'Python peak memory (MB, tracemalloc main process): {peak_mem_mb:.4f}\n')
        f.write('Note: tracemalloc reports Python-level peak allocations in the main process, not full process RSS and not child-process memory.\n')
        f.write('Speed note: this script uses CPU multiprocessing over threshold pairs. GPU acceleration is not used because the pipeline relies on SciPy sparse matrices.\n')


if __name__ == '__main__':
    main()
