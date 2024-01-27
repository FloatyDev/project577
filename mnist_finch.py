import numpy as np
import h5py

from sklearn.datasets import fetch_openml
from finch import FINCH
from sklearn.metrics import normalized_mutual_info_score as nmi_score
from sklearn.utils import resample
from utils.util import mean_confidence_interval, plot_nmi_scores

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
# Apply FINCH clustering
# CNN features
nmi_raw_scores = []
nmi_cnn_scores = []
number_of_bootstrap_samples = 20

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
    nmi_raw_scores.append(nmi_raw)
    nmi_cnn_scores.append(nmi_cnn)

# Calculate CI
ci, low, high = mean_confidence_interval(nmi_raw_scores)
ci_cnn, low_cnn, high_cnn = mean_confidence_interval(nmi_cnn_scores)


# Calculate mean and variance
iterations = range(1, len(nmi_raw_scores) + 1)

mean_nmi_raw = float(np.mean(nmi_raw_scores))
plot_nmi_scores(iterations, nmi_raw_scores, mean_nmi_raw, low, high)

mean_nmi_cnn = float(np.mean(nmi_cnn_scores))
plot_nmi_scores(iterations, nmi_cnn_scores, mean_nmi_cnn, low_cnn, high_cnn)
