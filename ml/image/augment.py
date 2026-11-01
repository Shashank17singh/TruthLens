"""Train and evaluation transforms. Every augmentation can be switched off for the ablation."""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def eval_transform(mean: tuple[float, ...], std: tuple[float, ...]) -> A.Compose:
    return A.Compose([A.Normalize(mean=mean, std=std), ToTensorV2()])


def train_transform(config: dict, size: int, mean: tuple[float, ...], std: tuple[float, ...], seed: int) -> A.Compose:
    """Geometry first, compression last, which is the order a shared image goes through."""
    steps: list[A.BasicTransform] = [A.HorizontalFlip(p=0.5)]
    if config["random_crop"]["enabled"]:
        steps.append(A.RandomResizedCrop((size, size), scale=tuple(config["random_crop"]["scale"]), ratio=(0.9, 1.1)))
    if config["color_jitter"]["enabled"]:
        jitter = config["color_jitter"]
        steps.append(A.ColorJitter(
            brightness=tuple(jitter["brightness"]),
            contrast=tuple(jitter["contrast"]),
            saturation=tuple(jitter["saturation"]),
            hue=tuple(jitter["hue"]),
            p=jitter["p"],
        ))
    if config["gaussian_blur"]["enabled"]:
        blur = config["gaussian_blur"]
        steps.append(A.GaussianBlur(sigma_limit=tuple(blur["sigma"]), p=blur["p"]))
    if config["resize"]["enabled"]:
        resize = config["resize"]
        steps.append(A.Downscale(scale_range=tuple(resize["scale"]), p=resize["p"]))
    if config["jpeg"]["enabled"]:
        jpeg = config["jpeg"]
        steps.append(A.ImageCompression(quality_range=tuple(jpeg["quality"]), p=jpeg["p"]))
    steps += [A.Normalize(mean=mean, std=std), ToTensorV2()]
    return A.Compose(steps, seed=seed)
