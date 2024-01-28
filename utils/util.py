import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import cv2
import os
import requests
from tqdm import tqdm
import shutil


def calculate_hog_features(mask):
    # Initialize the HOG descriptor
    mask = mask.astype(np.uint8) * 255

    winSize = (
        mask.shape[1] // 8 * 8,
        mask.shape[0] // 8 * 8,
    )  # Ensure divisibility by 8
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

    # Compute HOG features
    hog_features = hog.compute(mask)

    # Reshape to 1D array
    hog_features = np.array(hog_features).flatten()

    return hog_features


def prepare_folder(folder_path):
    """
    Prepare a folder by wiping its contents if it exists, or creating it if it does not.

    Args:
    folder_path (str): The path to the folder.
    """
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Wipe the folder contents
        shutil.rmtree(folder_path)

    # Create the folder
    os.makedirs(folder_path, exist_ok=True)
    print(f"Folder '{folder_path}' is ready.")


def download_weights(file_path, url):
    """
    Download the file from the specified URL if it doesn't exist, with a progress bar.

    Args:
    file_path (str): The path where the file should be saved.
    url (str): The URL to download the file from.
    """
    # Check if the file already exists
    if not os.path.exists(file_path):
        print(f"Downloading weights to {file_path}...")

        # Send a HTTP request to the server
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Ensure the request was successful

        total_size_in_bytes = int(response.headers.get("content-length", 0))
        chunk_size = 1024  # 1 Kilobyte

        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

        # Write the file to the specified path in chunks
        with open(file_path, "wb") as file:
            for data in response.iter_content(chunk_size):
                progress_bar.update(len(data))
                file.write(data)

        progress_bar.close()

        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")
        else:
            print("Download complete.")
    else:
        print("Weights file already exists.")


def mean_confidence_interval(data, confidence=0.95):
    """
    Calculate the mean and the confidence interval of the given data.

    Args:
    data (list or array-like): The dataset for which the mean and confidence interval are to be calculated.
    confidence (float): The confidence level for the interval. Default is 0.95 for a 95% confidence interval.

    Returns:
    tuple: A tuple containing the mean, lower bound of the confidence interval, and upper bound of the confidence interval.
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, m - h, m + h


def plot_scores(scores, mean, low, high, filename):
    """
    Plot the scores over iterations with a confidence interval and save the plot to a file.

    Args:
    scores (list or array-like): The scores to be plotted.
    mean (float): The mean score.
    low (float): The lower bound of the confidence interval.
    high (float): The upper bound of the confidence interval.
    filename (str): The filename where the plot will be saved.

    """
    iterations = range(1, len(scores) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, scores, label="Score per Iteration")
    plt.axhline(y=mean, color="r", linestyle="-", label=f"Mean Score = {mean:.3f}")
    plt.fill_between(
        iterations, low, high, color="gray", alpha=0.2, label="95% Confidence Interval"
    )
    plt.xlabel("Bootstrap Iteration")
    plt.ylabel("Score")
    plt.title(f'Scores Over Iterations for {filename.split("_")[0].upper()}')
    plt.legend()

    plt.savefig(filename)
    plt.close()


def display_cluster_masks(masks, clusters, partition_id):
    """
    Display and print information about the clustered masks for a given partition.

    Args:
    masks (list of dict): A list where each element is a dictionary containing the 'segmentation' key with mask data.
    clusters (numpy.ndarray): An array of cluster labels, where each row corresponds to a mask and columns to partitions.
    partition_id (int): The specific partition to consider for displaying the clusters.

    """
    unique_clusters = np.unique(clusters[:, partition_id])
    print(f"There are {len(unique_clusters)} unique clusters")

    for cluster_id in unique_clusters:
        # Get the indices for the current cluster
        cluster_indices = np.where(clusters[:, partition_id] == cluster_id)[0]

        print(f"cluster id {cluster_id} has {cluster_indices.shape} shape")
        # Retrieve masks for these indices
        clustered_masks = [masks[i]["segmentation"] for i in cluster_indices]
        print(f"shape of clustered_masks is {np.array(clustered_masks).shape}")
        # Now use the display function similar to the one provided earlier
        display_cluster_masks_id(clustered_masks, cluster_id)


def display_cluster_masks_id(clustered_masks, cluster_id):
    """
    Display and save the clustered masks for a given cluster ID.

    Args:
    clustered_masks (list of numpy arrays): A list of mask arrays belonging to the same cluster.
    cluster_id (int): The ID of the cluster these masks belong to.

    """
    num_masks = len(clustered_masks)
    cols = min(num_masks, 4)  # Display up to 4 masks per row
    rows = num_masks // cols + (num_masks % cols > 0)

    _, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axs = axs.ravel() if num_masks > 1 else [axs]

    for i, mask in enumerate(clustered_masks):
        axs[i].imshow(mask, cmap="gray")
        axs[i].axis("off")
        axs[i].set_title(f"Cluster {cluster_id}")

    # Turn off any unused subplots
    for i in range(num_masks, rows * cols):
        axs[i].axis("off")
    # Ensure the directory exists

    plt.savefig(f"./sam_finch_results/cluster_{cluster_id}.png")
    plt.close()


def calculate_hu_moments(mask):
    """
    Calculate the Hu Moments for a given mask.

    Args:
    mask (numpy array): The mask for which to calculate the Hu Moments.

    Returns:
    numpy array: A 1D array of the 7 Hu Moments for the mask.

    Description:
    This function calculates the Hu Moments, which are shape descriptors invariant to translation, scale, and rotation.
    The function also applies a logarithmic transformation to these moments for normalization purposes.
    """
    # Convert the boolean mask to uint8
    mask = mask.astype(np.uint8) * 255

    moments = cv2.moments(mask)
    huMoments = cv2.HuMoments(moments)
    # Log scale hu moments
    for i in range(0, 7):
        huMoments[i] = -1 * np.copysign(1.0, huMoments[i]) * np.log10(abs(huMoments[i]))
    return huMoments.flatten()


# Function for annotations
def show_anns(anns):
    """
    Display annotations on an image.

    Args:
    anns (list of dictionaries): A list of annotations, where each annotation is a dictionary containing the 'segmentation'
    key with mask data and the 'area' key with the area of the mask.

    Description:
    This function sorts the annotations by area and displays them overlaid on a blank image. Each annotation is shown
    in a different color. The annotations are sorted so that larger annotations are displayed first.
    """
    if len(anns) == 0:
        return

    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
