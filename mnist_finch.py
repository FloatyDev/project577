import numpy as np
import h5py

from sklearn.datasets import fetch_openml
from finch import FINCH
from sklearn.metrics import normalized_mutual_info_score as nmi_score
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
)
from sklearn.utils import resample
from utils.util import mean_confidence_interval, plot_scores

# Load Raw MNIST dataset
# --------------------------------------------------
mnist_data, mnist_labels = fetch_openml("mnist_784", version=1, return_X_y=True)
data = mnist_data / 255.0  # Normalize data
labels = mnist_labels
labels = labels.astype(np.int32)
X = data[:10000]
y = labels[:10000]
# --------------------------------------------------

# Load CNN features
# --------------------------------------------------
f_cnndata = h5py.File("./data/mnist10k/data.mat", "r")
cnn_data = f_cnndata.get("data")

# Have to transpose because Matlab is column-major and Python is row-major
cnn_data = np.array(cnn_data).T

# Load labels
f_cnnlabel = h5py.File("./data/mnist10k/labels.mat", "r")

ground_truth_cnn = f_cnnlabel.get("labels")
ground_truth_cnn = np.array(ground_truth_cnn)

# squeeze to make 1D
ground_truth_cnn = np.squeeze(np.array(ground_truth_cnn))
print(f"ground_truth, shape : {ground_truth_cnn}")
# --------------------------------------------------


# Apply FINCH clustering
# --------------------------------------------------
print("FINCH clustering results for MNIST (automatic clustering):")
nmi_raw_scores, nmi_cnn_scores = [], []
nmi_raw_scores, nmi_cnn_scores = [], []
silhouette_raw_scores, silhouette_cnn_scores = [], []
davies_bouldin_raw_scores, davies_bouldin_cnn_scores = [], []
calinski_harabasz_raw_scores, calinski_harabasz_cnn_scores = [], []
adjusted_rand_raw_scores, adjusted_rand_cnn_scores = [], []
number_of_bootstrap_samples = 20

# Bootstrapping to identify more precise metrics
for _ in range(number_of_bootstrap_samples):
    random_seed = np.random.randint(0, 1000)

    X_raw_resampled = resample(X, n_samples=X.shape[0], random_state=random_seed)
    y_raw_resampled = resample(y, n_samples=len(y), random_state=random_seed)

    X_cnn_resampled = resample(
        cnn_data, n_samples=cnn_data.shape[0], random_state=random_seed
    )
    y_cnn_resampled = resample(
        ground_truth_cnn, n_samples=len(ground_truth_cnn), random_state=random_seed
    )

    clusters, num_clust, req_c = FINCH(X_raw_resampled, req_clust=10, verbose=False)
    clusters_cnn, num_clust_cnn, req_c_cnn = FINCH(
        X_cnn_resampled, req_clust=10, verbose=False
    )
    # Calculate NMI score
    nmi_raw = nmi_score(y_raw_resampled, req_c)
    nmi_cnn = nmi_score(y_cnn_resampled, req_c_cnn)
    # Calculate Silhouette Score
    silhouette_raw = silhouette_score(X_raw_resampled, req_c)
    silhouette_cnn = silhouette_score(X_cnn_resampled, req_c_cnn)

    # Calculate Davies-Bouldin Score
    davies_bouldin_raw = davies_bouldin_score(X_raw_resampled, req_c)
    davies_bouldin_cnn = davies_bouldin_score(X_cnn_resampled, req_c_cnn)

    # Calculate Calinski-Harabasz Index
    calinski_harabasz_raw = calinski_harabasz_score(X_raw_resampled, req_c)
    calinski_harabasz_cnn = calinski_harabasz_score(X_cnn_resampled, req_c_cnn)

    # Calculate Adjusted Rand Index
    adjusted_rand_raw = adjusted_rand_score(y_raw_resampled, req_c)
    adjusted_rand_cnn = adjusted_rand_score(y_cnn_resampled, req_c_cnn)

    nmi_raw_scores.append(nmi_raw)
    nmi_cnn_scores.append(nmi_cnn)
    silhouette_raw_scores.append(silhouette_raw)
    silhouette_cnn_scores.append(silhouette_cnn)
    davies_bouldin_raw_scores.append(davies_bouldin_raw)
    davies_bouldin_cnn_scores.append(davies_bouldin_cnn)
    calinski_harabasz_raw_scores.append(calinski_harabasz_raw)
    calinski_harabasz_cnn_scores.append(calinski_harabasz_cnn)
    adjusted_rand_raw_scores.append(adjusted_rand_raw)
    adjusted_rand_cnn_scores.append(adjusted_rand_cnn)
# Calculate CI
ci, low, high = mean_confidence_interval(nmi_raw_scores)
ci_cnn, low_cnn, high_cnn = mean_confidence_interval(nmi_cnn_scores)

# Silhouette Scores
ci_silhouette_raw, low_silhouette_raw, high_silhouette_raw = mean_confidence_interval(
    silhouette_raw_scores
)
ci_silhouette_cnn, low_silhouette_cnn, high_silhouette_cnn = mean_confidence_interval(
    silhouette_cnn_scores
)

# Davies-Bouldin Scores
(
    ci_davies_bouldin_raw,
    low_davies_bouldin_raw,
    high_davies_bouldin_raw,
) = mean_confidence_interval(davies_bouldin_raw_scores)
(
    ci_davies_bouldin_cnn,
    low_davies_bouldin_cnn,
    high_davies_bouldin_cnn,
) = mean_confidence_interval(davies_bouldin_cnn_scores)

# Calinski-Harabasz Scores
(
    ci_calinski_harabasz_raw,
    low_calinski_harabasz_raw,
    high_calinski_harabasz_raw,
) = mean_confidence_interval(calinski_harabasz_raw_scores)
(
    ci_calinski_harabasz_cnn,
    low_calinski_harabasz_cnn,
    high_calinski_harabasz_cnn,
) = mean_confidence_interval(calinski_harabasz_cnn_scores)

# Adjusted Rand Index Scores
(
    ci_adjusted_rand_raw,
    low_adjusted_rand_raw,
    high_adjusted_rand_raw,
) = mean_confidence_interval(adjusted_rand_raw_scores)
(
    ci_adjusted_rand_cnn,
    low_adjusted_rand_cnn,
    high_adjusted_rand_cnn,
) = mean_confidence_interval(adjusted_rand_cnn_scores)


mean_nmi_raw = float(np.mean(nmi_raw_scores))
plot_scores(nmi_raw_scores, mean_nmi_raw, low, high, "raw_nmi.png")

mean_nmi_cnn = float(np.mean(nmi_cnn_scores))
plot_scores(nmi_cnn_scores, mean_nmi_cnn, low_cnn, high_cnn, "cnn_nmi.png")

# Silhouette Scores
mean_silhouette_raw = float(np.mean(silhouette_raw_scores))
plot_scores(
    silhouette_raw_scores,
    mean_silhouette_raw,
    low_silhouette_raw,
    high_silhouette_raw,
    "raw_silhouette.png",
)
mean_silhouette_cnn = float(np.mean(silhouette_cnn_scores))
plot_scores(
    silhouette_cnn_scores,
    mean_silhouette_cnn,
    low_silhouette_cnn,
    high_silhouette_cnn,
    "cnn_silhouette.png",
)

# Davies-Bouldin Scores
mean_davies_bouldin_raw = float(np.mean(davies_bouldin_raw_scores))
plot_scores(
    davies_bouldin_raw_scores,
    mean_davies_bouldin_raw,
    low_davies_bouldin_raw,
    high_davies_bouldin_raw,
    "raw_davies_bouldin.png",
)
mean_davies_bouldin_cnn = float(np.mean(davies_bouldin_cnn_scores))
plot_scores(
    davies_bouldin_cnn_scores,
    mean_davies_bouldin_cnn,
    low_davies_bouldin_cnn,
    high_davies_bouldin_cnn,
    "cnn_davies_bouldin.png",
)

# Calinski-Harabasz Scores
mean_calinski_harabasz_raw = float(np.mean(calinski_harabasz_raw_scores))
plot_scores(
    calinski_harabasz_raw_scores,
    mean_calinski_harabasz_raw,
    low_calinski_harabasz_raw,
    high_calinski_harabasz_raw,
    "raw_calinski_harabasz.png",
)
mean_calinski_harabasz_cnn = float(np.mean(calinski_harabasz_cnn_scores))
plot_scores(
    calinski_harabasz_cnn_scores,
    mean_calinski_harabasz_cnn,
    low_calinski_harabasz_cnn,
    high_calinski_harabasz_cnn,
    "cnn_calinski_harabasz.png",
)

# Adjusted Rand Index Scores
mean_adjusted_rand_raw = float(np.mean(adjusted_rand_raw_scores))
plot_scores(
    adjusted_rand_raw_scores,
    mean_adjusted_rand_raw,
    low_adjusted_rand_raw,
    high_adjusted_rand_raw,
    "raw_adjusted_rand.png",
)
mean_adjusted_rand_cnn = float(np.mean(adjusted_rand_cnn_scores))
plot_scores(
    adjusted_rand_cnn_scores,
    mean_adjusted_rand_cnn,
    low_adjusted_rand_cnn,
    high_adjusted_rand_cnn,
    "cnn_adjusted_rand.png",
)
