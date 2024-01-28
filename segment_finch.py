import cv2
import numpy as np
import numpy as np
import torch
import cv2
import sys
import matplotlib.pyplot as plt

sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator


def calculate_hu_moments(mask):
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


def main():
    # Extract Hu Moments for each mask

    image = cv2.imread("./images/truck.jpg")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Preparing SAM model
    sam_checkpoint = "./sam_weights/sam_vit_h_4b8939.pth"

    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=0)  # tried 0 better than default

    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image=image)
    hu_moments_list = []
    for i, mask in enumerate(masks):
        mask_image = (mask["segmentation"] * 255).astype(
            np.uint8
        )  # Convert to uint8 format
        hu_moments_list.append(calculate_hu_moments(mask["segmentation"]))
        cv2.imwrite(f"./masks/mask_{i}.png", mask_image)

    # Stack the Hu Moments vectors into a matrix
    print(f"length of hu_moments_list: {len(hu_moments_list)}")
    print(hu_moments_list[0].shape)
    print(hu_moments_list[0])
    hu_moments_matrix = np.vstack(hu_moments_list)
    print(f"shape of matrix: {hu_moments_matrix.shape}")

    # save segmented image
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    plt.axis("off")
    show_anns(masks)
    plt.suptitle(
        "Default Settings SAM (automatic mask)", fontsize=34, fontweight="bold"
    )
    plt.savefig("sam.png")
    # features = np.array([calculate_hu_moments(mask) for mask in masks])


if __name__ == "__main__":
    main()
