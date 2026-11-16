"""Train one audio detector.

    python -m ml.audio.train model=lcnn feature=lfcc
    python -m ml.audio.train -m model=resnet18,lcnn feature=mel,lfcc,cqt
    python -m ml.audio.train model=wav2vec2
    python -m ml.audio.train augment=none

Re-running the same command resumes from the last epoch.
"""

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from ml.audio.data import eval_loader, train_loader
from ml.audio.model import build_model, checkpoint_metadata, unfreeze_schedule
from ml.common.manifest import read_manifest
from ml.common.seed import set_seed
from ml.common.training import fit


def run_name(config: dict) -> str:
    feature = "wave" if config["model"]["kind"] == "ssl" else config["feature"]["name"]
    return f"audio-{config['model']['name']}-{feature}-{config['augment']['name']}-{config['version']}"


def train(config: dict) -> Path:
    """Train until early stopping and return the path of the best checkpoint."""
    set_seed(config["seed"])
    frame = read_manifest(Path(config["manifest"]))
    root = Path(config["data_root"])
    model = build_model(config)
    batch_size = config["model"]["batch_size"]
    validation = frame[frame["split"] == "val"].reset_index(drop=True)
    return fit(
        model,
        train_loader(frame[frame["split"] == "train"], root, config, batch_size),
        eval_loader(validation, root, config, batch_size * 2),
        validation["label"].to_numpy(),
        config,
        run_name(config),
        checkpoint_metadata(model),
        before_epoch=unfreeze_schedule(config),
    )


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(config: DictConfig) -> None:
    print(train(OmegaConf.to_container(config, resolve=True)))


if __name__ == "__main__":
    main()
