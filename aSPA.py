import numpy as np
import pandas as pd

#-----------------------------------
# This implementation is heavily inspired (almost stolen) by the MultiHorizonSPA package R code for aSPA
#-----------------------------------


# -------------------------
# QS HAC variance (Andrews 1991) - ported
# -------------------------
def _qs_weights(x):
    argQS = 6 * np.pi * x / 5.0
    # prevent division by zero
    w1 = 3.0 / (argQS**2)
    w2 = (np.sin(argQS) / argQS) - np.cos(argQS)
    wQS = w1 * w2
    # handle small x (x->0) -> limit wQS -> 1
    wQS = np.where(np.isfinite(wQS), wQS, 1.0)
    return wQS

def qs_hac_variance(y):
    """
    y: (T,) or (T,N) array of mean-zero series. Returns variance estimates per column (length N).
    This mirrors the quadratic-spectral HAC used in the literature.
    """
    y = np.asarray(y)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    T, N = y.shape
    if T <= 1:
        return np.zeros(N)
    bw = 1.3 * (T ** (1/5.0))
    if bw < 1:
        bw = 1.0
    weights = _qs_weights(np.arange(1, T) / bw)  # length T-1
    out = np.zeros(N)
    ydemean = y - np.nanmean(y, axis=0)
    for i in range(N):
        v = ydemean[:, i]
        s0 = (v @ v) / T
        ssum = s0
        for j, wj in enumerate(weights, start=1):
            cov = (v[:T - j] @ v[j:]) / T
            ssum += 2.0 * wj * cov
        out[i] = ssum
    return out

# -------------------------
# MBB helpers
# -------------------------
def _get_mbb_indices(T, L, rng):
    """
    Generate MBB indices of length T, 0-based, by the approach used in Quaedvlieg/R implementation.
    rng: numpy.random.Generator
    """
    # start position uniformly 1..T -> convert to 0..T-1
    pos = int(np.ceil(T * rng.random())) - 1
    ids = np.empty(T, dtype=int)
    ids[0] = pos
    for t in range(1, T):
        # R's code: if rem(t, L)==0 (1-based) then draw new start else increment
        if (t + 1) % L == 0:
            pos = int(np.ceil(T * rng.random())) - 1
            ids[t] = pos
        else:
            pos = ids[t - 1] + 1
            if pos >= T:
                pos = 0
            ids[t] = pos
    return ids

def _mbb_variance(bsample, L):
    """
    bsample: (T,1) or (T,) demeaned series from bootstrap sample (constructed by indexing original demeaned series with MBB indices).
    Returns scalar variance estimate (matching R's MBB variance logic using block sums).
    The function below follows the MBB variance formula using first K*L observations where K=floor(T/L).
    """
    arr = np.asarray(bsample).reshape(-1, 1)
    T = arr.shape[0]
    K = T // L
    if K < 1:
        # fallback: sample variance
        return np.nanvar(arr, ddof=0)
    temp = arr[:K * L, :].reshape(K, L, 1)   # (K, L, 1)
    sums = temp.sum(axis=1)                  # (K, 1)
    omega = (sums ** 2).mean(axis=0) / L     # shape (1,)
    return omega[0]

# -------------------------
# aSPA t-stat and bootstrap
# -------------------------
def aSPA_test_from_lossdiff_matrix(LossDiff, weights=None, L=3, B=999, seed=None):
    """
    LossDiff: numpy array T x H (loss differential per time & horizon) where each column is dh_{i,j,t}.
              Must be full (no missing) for the T rows used — caller should prepare intersection of times.
    weights: length-H array; if None uses equal weights = 1/H
    L: block length (default 3)
    B: number of bootstrap resamples (default 999)
    seed: random seed for reproducibility
    Returns dict with:
      t_obs: observed t_aSPA
      p_value: one-sided p-value for H1: mu > 0 (small p => model better than benchmark)
      t_boot: array of bootstrap t statistics (length B)
      mu_hat: observed weighted mean
      omega_hat: HAC variance estimate
    """
    D = np.asarray(LossDiff, dtype=float)
    if D.ndim != 2:
        raise ValueError("LossDiff must be 2D (T x H).")
    T, H = D.shape
    if weights is None:
        weights = np.ones(H) / H
    else:
        weights = np.asarray(weights, dtype=float).reshape(H)
        weights = weights / weights.sum()

    # Weighted series (T,)
    wD = (D * weights.reshape(1, -1)).sum(axis=1)
    mu_hat = np.nanmean(wD)           # 1/T sum_t weighted average
    # HAC variance estimate of the weighted series
    omega_hat = qs_hac_variance(wD.reshape(-1, 1))[0]
    if omega_hat <= 0 or np.isnan(omega_hat):
        omega_hat = 1e-12

    t_obs = np.sqrt(T) * mu_hat / np.sqrt(omega_hat)

    # demeaned series (for bootstrap centering)
    demeaned = wD - mu_hat

    rng = np.random.default_rng(seed)
    t_boot = np.zeros(B)
    for b in range(B):
        ids = _get_mbb_indices(T, L, rng)
        b_sample = demeaned[ids]   # length T (repeated blocks)
        zeta_b = _mbb_variance(b_sample, L)
        if zeta_b is None or np.isnan(zeta_b) or zeta_b <= 0:
            zeta_b = 1e-12
        t_boot[b] = np.sqrt(T) * b_sample.mean() / np.sqrt(zeta_b)

    # one-sided p-value (H1: mu > 0)
    p_value = np.mean(t_boot >= t_obs)   # proportion of boot t >= observed
    return {
        "t_obs": float(t_obs),
        "p_value": float(p_value),
        "t_boot": t_boot,
        "mu_hat": float(mu_hat),
        "omega_hat": float(omega_hat),
        "T": int(T),
        "H": int(H)
    }

# -------------------------
# high-level wrapper that computes aSPA per country for model vs benchmark
# -------------------------
def compute_aSPA_per_country(model_name, benchmark_name, res_dict, true_df,
                             horizons=None, weights=None, L=3, B=999, seed=None,
                             require_all_horizons=False):
    """
    For each country, build the LossDiff matrix across horizons available for model and benchmark,
    align times so we obtain a common T across the horizons used, and run aSPA_test_from_lossdiff_matrix.
    Parameters:
      - model_name, benchmark_name: strings keys into res_dict
      - res_dict: dict of model -> {h -> DataFrame(index=time, columns=countries)}
      - true_df: DataFrame indexed by dates (will be converted to PeriodIndex('M') for safety)
      - horizons: iterable of horizons to consider (default 1..12)
      - weights: if None -> equal weights 1/H_here (H_here = number of horizons used for this country)
                 if provided, must be length-H where H is number of horizons you keep (you can pass np.repeat(1/4,4) etc.)
      - require_all_horizons: if True, require that *all* horizons in `horizons` be present for the country and share common times.
                              If False (default), we use the horizons that are available and which share at least one common date across them.
    Returns:
      - results: dict country -> aSPA result dict (p_value, t_obs, T, H, ...)
      - summary: dict with mean_p_value and share_p_lt_0.05 and n_countries
    """
    if horizons is None:
        horizons = list(range(1, 13))
    # normalize true_df index to monthly periods (so alignment is robust)
    tdf = true_df.copy()

    results = {}
    pvals = []

    for country in tdf.columns:
        # build per-horizon series of squared errors for model and benchmark
        se_model = {}
        se_bench = {}
        times_per_h = {}
        used_horizons = []
        for h in horizons:
            pred_m = res_dict.get(model_name, {}).get(h)
            pred_b = res_dict.get(benchmark_name, {}).get(h)
            if pred_m is None or pred_b is None:
                continue
            pm = pred_m.copy()
            pb = pred_b.copy()
            pm.index = pd.to_datetime(pm.index).to_period("M")
            try: 
                pb.index = pd.to_datetime(pb.index).to_period("M")
            except Exception as e:
                pass
            if country not in pm.columns or country not in pb.columns:
                continue
            # common times with true
            common = tdf.index.intersection(pm.index).intersection(pb.index)

            if len(common) == 0:
                continue
            ym = pm.loc[common, country].astype(float)
            yb = pb.loc[common, country].astype(float)
            y_true = tdf.loc[common, country].astype(float)
            se_model[h] = (ym - y_true) ** 2
            se_bench[h] = (yb - y_true) ** 2
            times_per_h[h] = set(common)
            used_horizons.append(h)

        if len(used_horizons) == 0:
            continue

        # find common times across the used horizons (intersection)
        common_times = None
        for h in used_horizons:
            if common_times is None:
                common_times = set(times_per_h[h])
            else:
                common_times &= times_per_h[h]
        if not common_times:
            # no single common time across all horizons; skip country (conservative)
            continue
        common_times = sorted(common_times)
        # Build LossDiff matrix T x H_used (rows ordered by common_times)
        H_used = len(used_horizons)
        LossDiff = np.full((len(common_times), H_used), np.nan)
        col_h = []
        for j, h in enumerate(used_horizons):
            s_m = se_model[h].reindex(common_times)
            s_b = se_bench[h].reindex(common_times)
            LossDiff[:, j] = (s_m - s_b).values
            col_h.append(h)

        # drop any rows with NaN (shouldn't be any if we used intersection but be safe)
        mask = ~np.isnan(LossDiff).any(axis=1)
        LossDiff = LossDiff[mask, :]
        T_here = LossDiff.shape[0]
        if T_here <= 1:
            continue

        # determine weights: if user passed explicit weights (length matching H_used) use them
        if weights is None:
            w = np.ones(LossDiff.shape[1]) / LossDiff.shape[1]
        else:
            w = np.asarray(weights, dtype=float)
            if w.size != LossDiff.shape[1]:
                # if user gave weights of length original horizons, try to pick the ones for used_horizons
                try:
                    w_full = np.asarray(weights)
                    # if weights provided as dict/horizon->weight you could implement mapping; for now raise
                    raise ValueError("Provided weights length does not match number of used horizons.")
                except Exception:
                    raise
            w = w / w.sum()

        # run aSPA
        out = aSPA_test_from_lossdiff_matrix(LossDiff, weights=w, L=L, B=B, seed=seed)
        results[country] = out
        pvals.append(out["p_value"])

    # summary
    pvals = 1 - np.array(pvals)
    if pvals.size == 0:
        print("no_pvals")
        summary = {"mean_p_value": None, "share_p_lt_0.05": None, "n_countries": 0}
    else:
        summary = {
            "mean_p_value": float(np.nanmean(pvals)),
            "share_p_lt_0.05": float(np.mean(pvals < 0.05)),
            "n_countries": int(pvals.size)
        }

    return results, summary
