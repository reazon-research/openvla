## Prismatic Repository

### Directory Structure

- **Makefile**: Self-explanatory, work in progress.
- **README.md**: Project documentation.
- **pyproject.toml**: Centralizes configuration for building the project, specifying dependencies and tools.
- **requirements-min.txt**: Minimum requirements for the project.

### prismatic/

Contains miscellaneous setup for models, configs, processing, and training. Should not require editing.

### scripts/
- **generate.py**: Simple CLI script for interactive testing of generating from a pretrained VLM. Provides a minimal REPL for specifying image URLs, prompts, and language generation parameters.
- **preprocess.py**: Core script for automatically downloading raw VLM pretraining datasets, including LLaVA v1.5 datasets and stages. Runs download and extraction automatically.
- **pretrain.py**: Pretraining script for Prismatic VLM in native PyTorch, using Fully-Sharded Data Parallel (FSDP) for distributed training across GPUs. Assumes CUDA toolkit >= 11.0 for BF16 mixed precision.

#### additional-datasets/

- **lrv_instruct.py**: Preprocessing script for LRV-Instruct data. Requires prior download of images and JSON files.
- **lvis_instruct_4v.py**: Preprocessing script for LVIS-Instruct4v (language/chat) data. Requires prior download of images.

#### extern/

- **convert_openvla_weights_to_hf.py**: Converts weights from a Prismatic VLM model to HuggingFace format for compatibility with the transformers library.
- **verify_openvla.py**: Verifies functionality of a Prismatic model exported to HuggingFace using AutoClasses.

### vla-scripts/

- **deploy.py**: Sets up a server for deploying OpenVLA models via a REST API.
- **finetune.py**: Fine-tunes a model given a finetune model, dataset directory, name of fine-tuning dataset, and hyperparameters.
- **train.py**: Configures and automates training for OpenVLA models.

#### extern/

- **convert_openvla_weights_to_hf.py**: Utility script for converting OpenVLA weights to HuggingFace-compatible weights.
- **verify_openvla.py**: Checks functionality of a HuggingFace-exported OpenVLA model.

