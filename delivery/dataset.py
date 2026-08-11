"""
H.E.M.A. — Dataset utilities.

Images are pre-cropped conjunctiva strips on a transparent (RGBA) background,
with a highly elongated, non-square aspect ratio. Two design decisions follow
directly from that:

1. Composite onto a fixed black background before anything else, so the model
   sees a consistent 3-channel RGB image (transparent regions carry no clinical
   signal — they're just crop padding).
2. Resize preserving aspect ratio, then letterbox-pad to a square, rather than
   a naive square resize — a naive resize would squash the crescent shape and
   distort the very color/texture patterns the model needs to learn from.
"""
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def composite_and_letterbox(img: Image.Image, target_size: int = 224, bg=(0, 0, 0)) -> Image.Image:
    """Flatten RGBA onto a solid background, then resize+pad to a square, aspect-ratio preserved."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    background = Image.new("RGB", img.size, bg)
    background.paste(img, mask=img.split()[3])  # use alpha channel as mask
    img = background

    w, h = img.size
    scale = target_size / max(w, h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    img = img.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("RGB", (target_size, target_size), bg)
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    canvas.paste(img, (paste_x, paste_y))
    return canvas


class HemaConjunctivaDataset(Dataset):
    """Reads from a split folder of the form <root>/<Anemic|Non-anemic>/*.png"""

    LABEL_MAP = {"Non-anemic": 0, "Anemic": 1}

    def __init__(self, root_dir: str, target_size: int = 224, augment: bool = False):
        self.samples = []
        for label_name, label_idx in self.LABEL_MAP.items():
            folder = os.path.join(root_dir, label_name)
            if not os.path.isdir(folder):
                continue
            for fname in sorted(os.listdir(folder)):
                if fname.lower().endswith(".png"):
                    self.samples.append((os.path.join(folder, fname), label_idx, fname))

        self.target_size = target_size
        self.augment = augment

        aug_ops = []
        if augment:
            # Mild, clinically-plausible augmentation only: these are cropped tissue
            # patches, so we avoid aggressive color jitter that could mimic or mask
            # pallor/redness (the actual clinical signal).
            aug_ops = [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=10),
                T.ColorJitter(brightness=0.15, contrast=0.15),
            ]
        self.transform = T.Compose(aug_ops + [
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, fname = self.samples[idx]
        img = Image.open(path)
        img = composite_and_letterbox(img, self.target_size)
        img = self.transform(img)
        return img, label, fname

    def class_counts(self):
        counts = {0: 0, 1: 0}
        for _, label, _ in self.samples:
            counts[label] += 1
        return counts
