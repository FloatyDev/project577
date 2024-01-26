import numpy as np
import h5py
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from finch import FINCH
from sklearn.metrics import normalized_mutual_info_score as nmi_score
from sklearn.utils import resample
from utils.util import mean_confidence_interval

# Load Raw MNIST dataset
#--------------------------------------------------
mnist_data, mnist_labels = fetch_openml('mnist_784', version=1, return_X_y=True)
data = mnist_data / 255.0  # Normalize data
labels = mnist_labels
labels = labels.astype(np.int32)
print(f" label shape :{labels.shape}")
X = data[:10000]
y = labels[:10000]
#--------------------------------------------------

# Load CNN features
#--------------------------------------------------
f_cnndata = h5py.File('./data/mnist10k/data.mat', 'r')
cnn_data = f_cnndata.get('data')

# Have to transpose because Matlab is column-major and Python is row-major
cnn_data = np.array(cnn_data).T

# Load labels
f_cnnlabel = h5py.File('./data/mnist10k/labels.mat', 'r')

ground_truth_cnn = f_cnnlabel.get('labels')
ground_truth_cnn = np.array(ground_truth_cnn)

# squeeze to make 1D
ground_truth_cnn = np.squeeze(np.array(ground_truth_cnn))
print(f"ground_truth, shape : {ground_truth_cnn}")
#--------------------------------------------------



# Apply FINCH clustering
#--------------------------------------------------
print("FINCH clustering results for MNIST (automatic clustering):")
# Apply FINCH clustering
# CNN features
nmi_raw_scores = []
nmi_cnn_scores = []
number_of_bootstrap_samples = 10

# Set random state for reproducibility, but change that to np.random.randint(0, 1000) for real experiments
#print(f"X shape : {X.shape[0]} {len(y)} ")
for _ in range(number_of_bootstrap_samples):

    random_seed = np.random.randint(0, 1000)

    X_resampled = resample(X, n_samples= X.shape[0], random_state=random_seed)

    y_resampled = resample(y, n_samples=len(y), random_state=random_seed)

    clusters, num_clust, req_c = FINCH(X_resampled,req_clust=10,verbose=False)
    # Calculate NMI score
    nmi_raw = nmi_score(y_resampled, req_c)
    print(f"num_clust : {num_clust}")
    print(f"req_c : {np.unique(np.array(req_c))}")
    print(f"nmi_raw : {nmi_raw}")
    nmi_raw_scores.append(nmi_raw)

# Calculate CI
ci, low, high = mean_confidence_interval(nmi_raw_scores)
# Calculate mean and variance
mean_nmi = np.mean(nmi_raw_scores)
variance_nmi = np.var(nmi_raw_scores)
# Plotting
# Create histogram
plt.hist(nmi_raw_scores, bins=10, alpha=0.7, color='blue', label='NMI Scores')

# Add error bar for mean and standard deviation
plt.errorbar(mean_nmi, 15, xerr=variance_nmi, fmt='o', color='red', label='Mean ± Std Dev')

# Labels and title
plt.xlabel('NMI Score')
plt.ylabel('Frequency')
plt.title('Distribution of NMI Scores with Mean and Variance')
plt.legend()

plt.savefig('my_plot.png')  # Saves the plot to a file

#c1, num_clust1, req_c1 = FINCH(X, req_clust=10,verbose=True)
#--------------------------------------------------

# Compute NMI scores
#--------------------------------------------------
# Fifth column of c contains the cluster assignments for the 5th partition
# CNN scores
#score_cnn = nmi_score(ground_truth_cnn, c_cnn[:, 4])
#score_cnn_10_clusters = nmi_score(ground_truth_cnn, c_cnn_1[:, 4])
## Raw scores
#score_raw = nmi_score(y, c[:, 4])
#score_raw_1 = nmi_score(y, c[:, 3])
#score_raw_10_clusters = nmi_score(y, c1[:,4])
#--------------------------------------------------

# Print results
#--------------------------------------------------
#print("--------------------")
#print("Raw data (automatic clustering):")
#print("Number of clusters at each level:", num_clust1)
#print("Cluster assignments for the first few samples (Raw):\n", c1[:10])
#print('NMI Score (Raw) (6 clusters): {:.2f}'.format(score_raw_10_clusters* 100))
#print('NMI Score (Raw) (21 clusters): {:.2f}'.format(score_raw_10_clusters* 100))
#print("********************")
#print("Raw data (for 10 clusters):")
#print("Number of clusters at each level:", num_clust)
#print('NMI Score (Raw) (force 10 clusters): {:.2f}'.format(score_raw_10_clusters* 100))
#
#
#print("--------------------")
#print("CNN features results:")
#print("Number of clusters at each level:", num_clust_cnn)
#print("Cluster assignments for the first few samples (CNN):\n", c_cnn[:10])
#print('NMI Score (CNN) (10 clusters): {:.2f}'.format(score_cnn * 100))
#print("********************")
#print("CNN results (force 10 clusters):")
#print("Number of clusters at each level:", num_clust_cnn_1)
#print('NMI Score (CNN) (10 clusters): {:.2f}'.format(score_cnn_10_clusters * 100))
##--------------------------------------------------
