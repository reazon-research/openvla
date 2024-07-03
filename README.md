# OpenVLA: An Open-Source Vision-Language-Action Model

[![arXiv](https://img.shields.io/badge/arXiv-2406.09246-df2a2a.svg?style=for-the-badge)](https://arxiv.org/abs/2406.09246)
[![HF Models](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow?style=for-the-badge)](https://huggingface.co/openvla/openvla-7b)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/)
[![Python](https://img.shields.io/badge/python-3.10-blue?style=for-the-badge)](https://www.python.org)
[![License](https://img.shields.io/github/license/TRI-ML/prismatic-vlms?style=for-the-badge)](LICENSE)
 
[**Getting Started**](#getting-started) | [**Pretrained VLAs**](#pretrained-vlas) | [**Installation**](#installation) | 
[**Training VLAs from Scratch**](#training-vlas-from-scratch) | [**Project Website**](https://openvla.github.io/)

A simple and scalable codebase for training and fine-tuning vision-language-action models (VLAs) for generalist robotic 
manipulation:

- **Different Dataset Mixtures**: We natively support arbitrary datasets in RLDS format, including arbitrary mixtures of
  data from the [Open X-Embodiment Dataset](https://robotics-transformer-x.github.io/).
- **Easy Scaling**: Powered by PyTorch FSDP and Flash-Attention, we can quickly and efficiently train models from 1B - 
  34B parameters, with easily adaptable model architectures.
- **Native Fine-Tuning Support**: Built-in support (with examples) for various forms of fine-tuning (full, 
  partial, LoRA).

Built on top of [Prismatic VLMs](https://github.com/TRI-ML/prismatic-vlms).

## Getting Started

To get started with loading and running OpenVLA models for inference, we provide a lightweight interface that leverages
HuggingFace `transformers` AutoClasses, with minimal dependencies.

For example, to load `openvla-7b` for zero-shot instruction following in the
[BridgeData V2 environments](https://rail-berkeley.github.io/bridgedata/) with a Widow-X robot:

```python
# Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

import torch

# Load Processor & VLA
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", 
    attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
    trust_remote_code=True
    device_map="cuda:0" # must add this to avoid flash attention 2.0 and GPU initialization error
).to("cuda:0")

# Grab image input & format prompt
image: Image.Image = get_from_camera(...)
prompt = "In: What action should the robot take to {<INSTRUCTION>}?\nOut:"

# Predict Action (7-DoF; un-normalize for BridgeData V2)
inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

# Execute...
robot.act(action, ...)
```

We also provide an [example script for fine-tuning OpenVLA models for new tasks and 
embodiments](./vla-scripts/finetune.py); this script supports different fine-tuning modes -- including (quantized) 
low-rank adaptation (LoRA) supported by [HuggingFace's PEFT library](https://huggingface.co/docs/peft/en/index). 

For deployment, we provide a lightweight script for [serving OpenVLA models over a REST API](./vla-scripts/deploy.py), 
providing an easy way to integrate OpenVLA models into existing robot control stacks, 
removing any requirement for powerful on-device compute.

## Pretrained VLAs

We release two OpenVLA models trained as part of our work, with checkpoints, configs, and model cards available [on our
HuggingFace page](https://huggingface.co/openvla):
- [`openvla-7b`](https://huggingface.co/openvla/openvla-7b): The flagship model from our paper, trained from 
  the Prismatic `prism-dinosiglip-224px` VLM (based on a fused DINOv2 and SigLIP vision backbone, and Llama-2 LLM). 
  Trained on a large mixture of datasets from Open X-Embodiment spanning 970K trajectories 
  ([mixture details - see "Open-X Magic Soup++"](./prismatic/vla/datasets/rlds/oxe/mixtures.py)).
- [`openvla-v01-7b`](https://huggingface.co/openvla/openvla-7b-v01): An early model used during development, trained from
  the Prismatic `siglip-224px` VLM (singular SigLIP vision backbone, and a Vicuña v1.5 LLM). Trained on the same mixture
  of datasets as [Octo](https://github.com/octo-models/octo), but for significantly fewer GPU hours than our final model 
  ([mixture details - see "Open-X Magic Soup"](./prismatic/vla/datasets/rlds/oxe/mixtures.py)).

**Explicit Notes on Model Licensing & Commercial Use**: While all code in this repository is released under an MIT 
License, our pretrained models may inherit restrictions from the underlying base models we use. Specifically, both the
above models are derived from Llama-2, and as such are subject to the 
[Llama Community License](https://ai.meta.com/llama/license/).

---

## Installation

> **Note**: These installation instructions are for full-scale pretraining (and distributed fine-tuning); if looking to
  just run inference with OpenVLA models (or perform lightweight fine-tuning), see instructions above!

This repository was built using Python 3.10, but should be backwards compatible with any Python >= 3.8. We require
PyTorch 2.2.* -- installation instructions [can be found here](https://pytorch.org/get-started/locally/). The latest 
version of this repository was developed and thoroughly tested with:
  - PyTorch 2.2.0, torchvision 0.17.0, transformers 4.40.1, tokenizers 0.19.1, timm 0.9.10, and flash-attn 2.5.5

**[5/21/24] Note**: Following reported regressions and breaking changes in later versions of `transformers`, `timm`, and
`tokenizers` we explicitly pin the above versions of the dependencies. We are working on implementing thorough tests, 
and plan on relaxing these constraints as soon as we can.

Once PyTorch has been properly installed, you can install this package locally via an editable installation (or via 
`pip install git+https://github.com/openvla/openvla`):

```bash
cd openvla
pip install -e .

# Training additionally requires Flash-Attention 2 (https://github.com/Dao-AILab/flash-attention)
pip install packaging ninja

# Verify Ninja --> should return exit code "0"
ninja --version; echo $?

# Install Flash Attention 2
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install "flash-attn==2.5.5" --no-build-isolation
```

If you run into any problems during the installation process, please file a GitHub Issue.

**Note:** See `vla-scripts/` for full training and verification scripts for OpenVLA models. Note that `scripts/` is
mostly a holdover from the original (base) `prismatic-vlms` repository, with support for training and evaluating
visually-conditioned language models; while you can use this repo to train VLMs AND VLAs, note that trying to generate
language (via `scripts/generate.py`) with existing OpenVLA models will not work (as we only train current OpenVLA models
to generate actions, and actions alone).

## Training VLAs from Scratch

We provide full instructions and configurations for training OpenVLA models on (arbitrary subsets of) the
[Open X-Embodiment (OXE) Dataset](https://robotics-transformer-x.github.io/). If you run in to any issues with 
the following, see [VLA Troubleshooting](#vla-troubleshooting) below (or file a GitHub Issue).

### VLA Pretraining Datasets

We download and preprocess individual datasets from Open X-Embodiment in [RLDS format](https://github.com/google-research/rlds) following 
[this custom script](https://github.com/moojink/rlds_dataset_mod/blob/main/prepare_open_x.sh). See 
[mixtures.py](./prismatic/vla/datasets/rlds/oxe/mixtures.py) for the full list of component datasets (and mixture 
weights) we use to train `openvla-7b`. 
- **Important**: For the BridgeData V2 component, the version in OXE is out of date (as of 12/20/2023). Instead,
  you should download the dataset from the [official website](https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/) and place it under the subdirectory `bridge_orig/`. 
  Replace any reference to `bridge` in the OXE code with `bridge_orig`.

### VLA Configuration & Training Script

The entry point for VLA training is [`vla-scripts/train.py`](vla-scripts/train.py). We use 
[`draccus`](https://pypi.org/project/draccus) to provide a modular, dataclass-based interface for specifying VLA 
training configurations; existing VLA configurations are in [`prismatic/conf/vla.py`](prismatic/conf/vla.py). You can 
add your own training configuration and refer to it using the `--vla.type` command line argument.

We use PyTorch Fully Sharded Data Parallel (FSDP) to distribute training across GPUs. Launch training via `torchrun`:

```bash
# Train VLA on BridgeData V2 with the Prismatic DINO-SigLIP 224px Backbone on a Single Node (w/ 8 GPUs)
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train.py \
  --vla.type "prism-dinosiglip-224px+mx-bridge" \
  --data_root_dir <PATH TO OXE DATA ROOT> \
  --run_root_dir <PATH TO LOG/CHECKPOINT ROOT> \
  --wandb_project "<PROJECT>" \
  --wandb_entity "<ENTITY>"
```

### VLA Troubleshooting

The following are a list of known problems and corresponding fixes:

```bash
FileNotFoundError: Failed to construct dataset "fractal20220817_data", builder_kwargs "{'data_dir': '/path/to/processed/datasets/'}": Could not load dataset info from fractal20220817_data/0.1.0/dataset_info.json
```
- **Fix**: Downgrade `tensorflow-datasets` via `pip install tensorflow-datasets==4.9.3`.


```bash
AttributeError: 'DLataset' object has no attribute 'traj_map'. Did you mean: 'flat_map'?
```
- **Fix**: Upgrade `dlimp` to the newest version. You may have to `--force-reinstall` like so:
`pip install --no-deps --force-reinstall git+https://github.com/kvablack/dlimp@5edaa4691567873d495633f2708982b42edf1972`

---

## Repository Structure

High-level overview of repository/project file-tree:

+ `prismatic` - Package source; provides core utilities for model loading, training, data preprocessing, etc.
+ `scripts/` - Various scripts for data preprocessing, dataset management, and model interaction.
+ `vla-scripts/` - Core scripts for training, fine-tuning, and deploying VLAs.
+ `LICENSE` - All code is made available under the MIT License; happy hacking!
+ `Makefile` - Top-level Makefile (by default, supports linting - checking & auto-fix); extend as needed.
+ `pyproject.toml` - Full project configuration details (including dependencies), as well as tool configurations.
+ `README.md` - You are here!

---

#### Citation

If you find our code or models useful in your work, please cite [our paper](https://arxiv.org/abs/2406.09246):

```bibtex
@article{kim24openvla,
    title={OpenVLA: An Open-Source Vision-Language-Action Model},
    author={{Moo Jin} Kim and Karl Pertsch and Siddharth Karamcheti and Ted Xiao and Ashwin Balakrishna and Suraj Nair and Rafael Rafailov and Ethan Foster and Grace Lam and Pannag Sanketi and Quan Vuong and Thomas Kollar and Benjamin Burchfiel and Russ Tedrake and Dorsa Sadigh and Sergey Levine and Percy Liang and Chelsea Finn},
    journal = {arXiv preprint arXiv:2406.09246},
    year={2024}
} 
```
