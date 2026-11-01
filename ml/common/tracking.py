"""Experiment logging. Metrics always reach a local file, and W&B is added when enabled."""

import json
from pathlib import Path


class Tracker:
    def __init__(self, directory: Path, name: str, config: dict, kind: str, project: str):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "metrics.jsonl"
        self.run = None
        (directory / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        if kind == "wandb":
            import wandb

            self.run = wandb.init(project=project, name=name, id=name, resume="allow", config=config)

    def log(self, metrics: dict, step: int) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"step": step, **metrics}) + "\n")
        if self.run:
            self.run.log(metrics, step=step)

    def finish(self) -> None:
        if self.run:
            self.run.finish()
