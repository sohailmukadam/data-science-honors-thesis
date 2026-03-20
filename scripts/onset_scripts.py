import numpy as np

def sequence_overlap(X, season_length, nday):
    """
    Generate overlapping sequences of length `nday` from seasonal data.

    Parameters:
    X (ndarray): Input array of shape (n_rows, n_vars)
    season_length (int): Number of time steps per season/year
    nday (int): Length of sliding window

    Returns:
    Y (ndarray): Array of overlapping windows with shape (n_windows * n_years, n_vars * nday)

    Notes:
    - Pads the beginning of each season to allow full window construction.
    - Used for constructing wet/dry window indices.
    """
    n_rows, n_vars = np.shape(X)
    n_years = n_rows // season_length
    indice = []
    for i in range(nday):
        row = np.arange(i, season_length + i)
        indice.append(row)
    indice = np.array(indice)
    n_offsets, n_windows = indice.shape
    Y = np.zeros((n_windows * n_years, n_vars * nday))
    for i in range(n_years):
        sample = X[i * season_length : (i + 1) * season_length]
        sample = np.vstack([np.tile(sample[0], (nday - 1, 1)), sample])
        sample1 = np.zeros((n_windows, nday * n_vars))
        for j in range(nday):
            sample1[:, j*n_vars:(j+1)*n_vars] = sample[indice[j], :]
        Y[i * n_windows : (i + 1) * n_windows, :] = sample1
    return Y

def detect_rainy_season_onset(X, season_length, default_dry_threshold, wet_window_len, wet_threshold, dry_window_len, dry_threshold, confirm_window_len):
    """
    Detect rainy season onset using wet and dry window criteria.

    Parameters:
    X (ndarray): Daily precipitation data, shape (time, n_vars)
    season_length (int): Number of days per year/season
    default_dry_threshold (float): Threshold to classify a day as wet
    wet_window_len (int): Length of wet window (days)
    wet_threshold (float or 0): Threshold for wet window sum (0 = auto-compute)
    dry_window_len (int): Length of dry window (days)
    dry_threshold (float): Threshold for dry window sum
    confirm_window_len (int): Length of confirmation window

    Returns:
    O1 (ndarray): First detected onset dates (n_years, n_vars)
    O2 (ndarray): Confirmed onset dates (n_years, n_vars)
    MWmean (ndarray): Mean wet threshold per variable

    Description:
    - Step 1: Classify wet/dry days using threshold.
    - Step 2: Compute rolling wet and dry window sums.
    - Step 3: Identify candidate onset (O1).
    - Step 4: Confirm onset (O2) if no dry spell follows.

    Notes:
    - Assumes input is ordered by year.
    - Missing values are not explicitly handled.
    """
    n_rows, n_vars = np.shape(X)
    n_years = n_rows // season_length
    W = np.zeros(np.shape(X))
    W[X > default_dry_threshold] = 1
    wet_window_indices = None
    if wet_window_len > 1:
        single_year_days = np.arange(season_length)[:, None]
        indices_per_window = sequence_overlap(single_year_days, season_length, wet_window_len)
        indices_per_window = np.transpose(indices_per_window[wet_window_len - 1 : season_length, :])
        indices_per_window = (
            indices_per_window.reshape((-1, 1), order='F') @ np.ones((1, n_vars))
            + np.ones(((season_length - (wet_window_len - 1)) * wet_window_len, 1)) @ 
            (np.arange(0, season_length * n_vars, season_length).reshape(1, -1))
        )
        wet_window_indices = indices_per_window.reshape((wet_window_len, n_vars * (season_length - (wet_window_len - 1))), order='F')
    dry_window_indices = None
    if dry_window_len > 1:
        single_year_days = np.arange(season_length)[:, None]
        indices_per_window = sequence_overlap(single_year_days, season_length, dry_window_len)
        indices_per_window = np.transpose(indices_per_window[dry_window_len - 1 : season_length, :])
        indices_per_window = (
            indices_per_window.reshape((-1, 1), order='F') @ np.ones((1, n_vars))
            + np.ones(((season_length - (dry_window_len - 1)) * dry_window_len, 1)) @ 
            (np.arange(0, season_length * n_vars, season_length).reshape(1, -1))
        )
        dry_window_indices = indices_per_window.reshape((dry_window_len, n_vars * (season_length - (dry_window_len - 1))), order='F')
    O1 = np.full((n_years, n_vars), np.nan)
    O2 = np.full((n_years, n_vars), np.nan)
    S = confirm_window_len - (dry_window_len - 1)
    S2 = sequence_overlap(np.transpose([np.arange(season_length)]), season_length, S)
    S2 = np.transpose(S2[S - 1:season_length])
    Lw = season_length - (wet_window_len - 1)
    SWmean = np.zeros((n_years * Lw, n_vars))
    for i in range(n_years):
        sample = X[(i * season_length): ((i + 1) * season_length), :]
        sample_flat = sample.ravel(order="F")
        if wet_window_len > 1:
            SWmean[(i * Lw):(Lw * (i + 1)), :] = np.reshape(np.sum(sample_flat[wet_window_indices.astype(int)], axis=0), (season_length - (wet_window_len - 1), n_vars), order="F")
        else:
            SWmean[(i * Lw):(Lw * (i + 1)), :] = sample
    MWmean = np.zeros(n_vars)
    for i in range(n_vars):
        MWmean[i] = np.mean(SWmean[SWmean[:, i] > default_dry_threshold, i])
    if wet_threshold == 0:
        wet_threshold = MWmean
    else:
        wet_threshold = np.array([wet_threshold] * n_vars)
    for i in range(n_years):
        sample = X[(i * season_length): ((i + 1) * season_length), :]
        wsample = W[(i * season_length): ((i + 1) * season_length), :]
        sample_flat = sample.ravel(order="F")
        SW = sample
        SD = sample
        if wet_window_len > 1:
            SW = np.reshape(np.sum(sample_flat[wet_window_indices.astype(int)], axis=0), (season_length - (wet_window_len - 1), n_vars), order="F")
        if dry_window_len > 1:
            SD = np.reshape(np.sum(sample_flat[dry_window_indices.astype(int)], axis=0), (season_length - (dry_window_len - 1), n_vars), order="F")
        nrw, ncw = np.shape(SW)
        nrd, ncd = np.shape(SD)
        for j in range(n_vars):
            SW_extension = np.concatenate([SW[:, j], np.ones(wet_window_len - 1) * SW[season_length - wet_window_len, j]])
            SD_extension = np.concatenate([SD[:, j], np.zeros(dry_window_len - 1)])
            tab = np.column_stack([sample[:, j], wsample[:, j], SW_extension, SD_extension])
            o1 = np.where((tab[:, 2] >= wet_threshold[j]) & (tab[:, 1] == 1))[0]
            D = tab[:, 3]
            D = np.transpose(D[S2.astype(int)])
            D = np.vstack([D, np.zeros((confirm_window_len - dry_window_len, S))])
            if o1.size > 0:
                O1[i, j] = o1[0]
                tab2 = D[o1, :]
                o2 = o1[np.min(tab2, axis=1) > dry_threshold]
                if o2.size > 0:
                    O2[i, j] = o2[0]
    return O1, O2, MWmean
