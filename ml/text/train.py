"""Fine-tune one transformer AI-text detector.

    python -m ml.text.train model=roberta
    python -m ml.text.train model=deberta steps_per_epoch=2000

Re-running the same command resumes from the last epoch.
"""

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from ml.common.manifest import read_manifest
from ml.common.seed import set_seed
from ml.common.training import fit
from ml.text.data import eval_loader, train_loader
from ml.text.model import build_model


def run_name(config: dict) -> str:
    return f"text-{config['model']['name']}-{config['version']}"


def train(config: dict) -> Path:
    """Train until early stopping and return the path of the best checkpoint."""
    set_seed(config["seed"])
    frame = read_manifest(Path(config["manifest"]))
    root = Path(config["data_root"])
    model, tokenizer = build_model(config)
    batch_size = config["model"]["batch_size"]
    validation = frame[frame["split"] == "val"].reset_index(drop=True)
    return fit(
        model,
        train_loader(frame[frame["split"] == "train"], root, tokenizer, config, batch_size),
        eval_loader(validation, root, tokenizer, config, batch_size * 2),
        validation["label"].to_numpy(),
        config,
        run_name(config),
        {},
    )


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(config: DictConfig) -> None:
    print(train(OmegaConf.to_container(config, resolve=True)))


if __name__ == "__main__":
    main()
