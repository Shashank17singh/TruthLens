# TruthLens

TruthLens detects AI-generated or manipulated images, audio and text, explains each decision, and runs as a web app and a Chrome extension. The work is intended for an IEEE conference paper.

Training runs on Colab or Kaggle GPUs. Serving runs on CPU.

## Design documents

- [Architecture and monorepo structure](docs/architecture.md)
- [Database schema](docs/schema.md)
- [API contract](docs/api.md)
- [Datasets, licences and sizes](docs/datasets.md)
- [Design decisions](docs/decisions.md)

The code is built one feature prompt at a time, from 2.1 to 2.11. Each prompt is tested and committed before the next one starts.
