import os
import cv2
import numpy as np
import json
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F


class PoseDS(Dataset):
    def __init__(self, img_path, annotation_path, img_size=(224, 224)):
        self.img_path = img_path
        self.img_size = img_size
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
            ]
        )
        with open(annotation_path, "r") as f:
            self.annotation = json.load(f)

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        annot = self.annotation[index]
        joints_vis = annot["joints_vis"]
        joints = annot["joints"]
        scale = annot["scale"]
        center = annot["center"]
        image_name = annot["image"]

        # Load image
        img_path = os.path.join(self.img_path, image_name)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_height, orig_width = image.shape[:2]

        # Resize kps
        keypoints = np.array(joints, dtype=np.float32)
        visibility = np.array(joints_vis, dtype=np.float32)
        scale_x = self.img_size[0] / orig_width
        scale_y = self.img_size[1] / orig_height
        keypoints[:, 0] *= scale_x
        keypoints[:, 1] *= scale_y

        # Resize image
        image = cv2.resize(image, self.img_size)

        # Apply transformations
        image = self.transform(image)  # Now a tensor with shape [C, H, W]

        # Generate heatmaps
        heatmaps = self._generate_gaussian_heatmaps(
            keypoints, visibility, self.img_size
        )

        return image, heatmaps, keypoints, visibility

    def _generate_gaussian_heatmaps(self, keypoints, visibility, image_size, sigma=3):
        h, w = image_size
        num_keypoints = keypoints.shape[0]
        heatmaps = np.zeros((num_keypoints, h, w), dtype=np.float32)

        for i, (kp, vis) in enumerate(zip(keypoints, visibility)):
            if vis == 0:
                continue  # Skip no visible kps
            x, y = kp
            heatmaps[i] = self._generate_gaussian(np.zeros((h, w)), x, y, sigma=sigma)

        return torch.from_numpy(heatmaps)

    def _generate_gaussian(self, t, x, y, sigma=10):
        h, w = t.shape
        tmp_size = sigma * 3
        x = int(x)
        y = int(y)

        # Top-left
        x1, y1 = int(x - tmp_size), int(y - tmp_size)
        # Bottom-right
        x2, y2 = int(x + tmp_size + 1), int(y + tmp_size + 1)

        if x1 >= w or y1 >= h or x2 < 0 or y2 < 0:
            return t  # if out of bounds, return zero heatmap

        size = 2 * tmp_size + 1
        tx = np.arange(0, size, 1, np.float32)
        ty = tx[:, np.newaxis]
        x0 = y0 = size // 2

        # Gaussian fx
        g = np.exp(-((tx - x0) ** 2 + (ty - y0) ** 2) / (2 * sigma**2))

        # Determine the bounds of the source Gaussian
        g_x_min, g_x_max = max(0, -x1), min(x2, w) - x1
        g_y_min, g_y_max = max(0, -y1), min(y2, h) - y1

        # Fix Image range
        img_x_min, img_x_max = max(0, x1), min(x2, w)
        img_y_min, img_y_max = max(0, y1), min(y2, h)

        t[img_y_min:img_y_max, img_x_min:img_x_max] = g[
            g_y_min:g_y_max, g_x_min:g_x_max
        ]

        return t


# Testing code
# if __name__ == "__main__":
#     img_path = "./data/images"
#     annotation_path = "./data/annot/train.json"
#     dataset = PoseDS(img_path, annotation_path, img_size=(224, 224))
#     # First obj of dataset
#     image, heatmaps, keypoints, visibility = dataset[0]
#     print("Image shape:", image.shape)  # [C, H, W]
#     print("Heatmaps shape:", heatmaps.shape)  # [num_keypoints, H, W]
#     print("Keypoints shape:", keypoints.shape)  # [num_keypoints, 2]
#     print("Visibility shape:", visibility.shape)  # [num_keypoints]
#     # Visualize the first heatmap
#     heatmap_np = heatmaps[0].numpy()
#     # heatmap
#     plt.imshow(heatmap_np, cmap="hot")
#     plt.title("Heatmap for Keypoint 0")
#     plt.show()
#     image_pil = F.to_pil_image(image)
#     image_np = np.array(image_pil)
#     plt.imshow(image_np)
#     for i, (x, y) in enumerate(keypoints):
#         if visibility[i]:
#             plt.scatter(x, y, s=3, marker="o", c="red")
#     plt.title("Image with Keypoints")
#     plt.show()
