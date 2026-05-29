import numpy as np
from scipy.signal import butter, filtfilt
from scipy.ndimage import median_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Filters
def hampel_filter(signal: np.ndarray, window_size: int = 100, n_sigma: float = 3.0) -> np.ndarray:
    """
    Remove spike outliers from signal.
    """
    signal = np.array(signal, dtype=float)
    k = 1.4826  # Scale factor for Gaussian distribution

    medians = median_filter(signal, size=window_size, mode='mirror')
    mads = k * median_filter(np.abs(signal - medians), size=window_size, mode='mirror')

    outliers = np.abs(signal - medians) > n_sigma * mads
    signal[outliers] = medians[outliers]

    return signal


def butterworth_filter(
    signal: np.ndarray,
    cutoff: float = 0.1,
    fs: float = 1.0,
    order: int = 6,
    filter_type: str = 'low'
) -> np.ndarray:
    """
    Smooth signal by removing high-frequency.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
    return filtfilt(b, a, signal)


def apply_pca(
    data: np.ndarray,
    n_components: int = 10
) -> np.ndarray:
    """
    Reduce the number of CSI features using PCA.
    """
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    pca = PCA(n_components=n_components)
    return pca.fit_transform(data_scaled)


# Main pipeline function
def apply_csi_filters(
    A: np.ndarray,
    use_pca: bool = False,
    n_components: int = 10,
    hampel_window: int = 100,
    hampel_sigma: float = 3.0,
    butterworth_cutoff: float = 0.1,
    butterworth_order: int = 6,
) -> np.ndarray:
    """
    Run the full filter pipeline on a batch of CSI amplitude data.
    """
    A = np.array(A, dtype=float)

    # Apply Hampel filter
    A = np.apply_along_axis(
        hampel_filter, axis=1, arr=A,
        window_size=hampel_window, n_sigma=hampel_sigma
    )

    # Apply Butterworth filter
    A = np.apply_along_axis(
        butterworth_filter, axis=1, arr=A,
        cutoff=butterworth_cutoff, order=butterworth_order
    )

    # Apply PCA (optional)
    if use_pca:
        batch_size = A.shape[0]
        pca_results = []
        for i in range(batch_size):
            # A[i] shape: (time_steps, features)
            pca_result = apply_pca(A[i], n_components=n_components)
            pca_results.append(pca_result)
        A = np.stack(pca_results, axis=0)

    return A