import cv2
import numpy as np
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
import argparse
from finch import FINCH
from utils.util import (
    display_cluster_masks,
    calculate_hu_moments,
    calculate_hog_features,
    download_weights,
    show_anns,
    prepare_folder,
)

sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


def main():
    parser = argparse.ArgumentParser(description="Process the algo argument.")
    parser.add_argument(
        "--algo",
        type=str,
        choices=["hu", "hog"],
        required=True,
        help='Feature extraction algorithm to use: "hu" or "hog"',
    )
    args = parser.parse_args()

    image = cv2.imread("./images/truck.jpg")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Preparing SAM model
    download_weights(
        "./sam_weights/sam_vit_h_4b8939.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    )
    prepare_folder("./masks/")
    prepare_folder("./sam_finch_results/")
    sam_checkpoint = "./sam_weights/sam_vit_h_4b8939.pth"

    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=0)  # tried 0 better than default

    mask_generator = SamAutomaticMaskGenerator(
        sam, pred_iou_thresh=0.94, min_mask_region_area=10
    )
    masks = mask_generator.generate(image=image)
    feature_list = []
    for i, mask in enumerate(masks):
        mask_image = (mask["segmentation"] * 255).astype(
            np.uint8
        )  # Convert to uint8 format
        cv2.imwrite(f"./masks/mask_{i}.png", mask_image)

    # Calculate mask features
    if args.algo == "hog":
        feature_list = [calculate_hog_features(mask["segmentation"]) for mask in masks]
    elif args.algo == "hu":
        feature_list = [calculate_hu_moments(mask["segmentation"]) for mask in masks]

    print(f"length of feature list: {len(feature_list)}")
    # Stack list of feature vectors into a matrix
    feature_matrix = np.vstack(feature_list)

    print(f"shape of feature matrix: {feature_matrix.shape}")

    clusters, num_clust, _ = FINCH(feature_matrix, distance="cosine", verbose=True)
    print(f"Clusters per partition: {num_clust}")

    # save segmented image
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    plt.axis("off")
    show_anns(masks)
    plt.suptitle(
        "Default Settings SAM (automatic mask)", fontsize=34, fontweight="bold"
    )
    plt.savefig("sam.png")

    display_cluster_masks(masks, clusters, -1)


if __name__ == "__main__":
    main()
