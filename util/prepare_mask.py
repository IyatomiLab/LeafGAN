import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm

from models.grad_cam import GradCAM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default=None, help="leaf GAN's dataset")
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.35,
        help="heatmap threshold",
    )
    parser.add_argument(
        "--pretrain_path",
        "-p",
        type=str,
        default=None,
        help="pretrain model path of LFLSeg ",
    )
    parser.add_argument("--image_size", "-i", type=int, help="size of image")
    args = parser.parse_args()
    return args


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, paths, image_size):
        self.paths = paths
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx])
        return self.transform(img)


def get_heatmap(model, loader, device):
    heatmaps = []
    for img in tqdm(loader):
        img = img.to(device)
        with torch.enable_grad():
            _ = model.forward(img)
            model.backward(idx=0)
        heatmap = model.generate(target_layer="layer4.2")
        heatmaps.append(heatmap)
    return heatmaps


def save_heatmap(heatmaps, paths, out_dir, image_size, threshold):
    for heatmap, p in zip(heatmaps, paths):
        mask = cv2.resize(heatmap, dsize=(image_size, image_size))
        mask = (mask >= threshold).astype(int)
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2) * 255
        mask_img = Image.fromarray(mask.astype(np.uint8))
        mask_img.save(out_dir / p.name)


def main():
    opt = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # setting LFLSeg Model
    segResNet = models.resnet101()
    num_ftrs = segResNet.fc.in_features
    segResNet.fc = nn.Linear(num_ftrs, 3)
    segResNet.load_state_dict(torch.load(opt.pretrain_path), strict=True)
    segResNet.to(device)
    segResNet.eval()

    netLFLSeg = GradCAM(model=segResNet)

    # setup mask data folder
    data_root = Path(opt.source)
    dataset_dirs = [p for p in data_root.glob("*") if "mask" not in str(p)]
    mask_dataset_dirs = []
    for data_dir in dataset_dirs:
        out_dir = data_root / f"{data_dir.name}_mask"
        out_dir.mkdir(exist_ok=True)
        mask_dataset_dirs.append(out_dir)

    # get_mask
    for source_dir, out_dir in zip(dataset_dirs, mask_dataset_dirs):
        print(f"##### {source_dir.name} #####")
        paths = list(source_dir.glob("*"))
        loader = torch.utils.data.DataLoader(
            MyDataset(paths, opt.image_size),
            shuffle=False,
            batch_size=1,
            num_workers=1,
        )
        heatmaps = get_heatmap(netLFLSeg, loader, device)
        save_heatmap(heatmaps, paths, out_dir, opt.image_size, opt.threshold)

    print("done")


if __name__ == "__main__":
    main()
