"""Train one image detector.

    python -m ml.image.train model=convnext_tiny use_frequency=true
    python -m ml.image.train -m model=efficientnet_b0,convnext_tiny,vit_b16 use_frequency=true,false
    python -m ml.image.train augment=none

Re-running the same command resumes from the last epoch, so a Colab session that
ends mid-training loses at most one epoch.
"""

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from ml.common.manifest import read_manifest
from ml.common.seed import set_seed
from ml.common.training import fit
from ml.image.augment import eval_transform, train_transform
from ml.image.data import eval_loader, train_loader
from ml.image.model import build_model


def run_name(config: dict) -> str:
    branch = "freq" if config["use_frequency"] else "rgb"
    return f"image-{config['model']['name']}-{branch}-{config['augment']['name']}-{config['version']}"


def train(config: dict) -> Path:
    """Train until early stopping and return the path of the best checkpoint."""
    set_seed(config["seed"])
    frame = read_manifest(Path(config["manifest"]))
    root = Path(config["data_root"])
    model = build_model(config)
    batch_size = config["model"]["batch_size"]

    loader = train_loader(
        frame[frame["split"] == "train"],
        root,
        train_transform(config["augment"], config["image_size"], model.mean, model.std, config["seed"]),
        batch_size,
        config["num_workers"],
        config["seed"],
    )
    validation = frame[frame["split"] == "val"].reset_index(drop=True)
    validation_loader = eval_loader(validation, root, eval_transform(model.mean, model.std), batch_size * 2, config["num_workers"])
    return fit(
        model, loader, validation_loader, validation["label"].to_numpy(), config, run_name(config),
        {"mean": model.mean, "std": model.std},
    )


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(config: DictConfig) -> None:
    print(train(OmegaConf.to_container(config, resolve=True)))


if __name__ == "__main__":
    main()
