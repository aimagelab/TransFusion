# TransFusion 🚀
TransFusion is the official codebase for the ICML 2025 paper:

**[What if your pre-train changes? Rebasin of Task Vectors](https://arxiv.org/abs/2505.22697)** 📝

TransFusion provides a principled and practical framework for permutation-based model fusion, transfer, and zero-shot evaluation in CLIP and Vision Transformer (ViT) architectures. It enables research and applications in model merging, transfer learning, and robust evaluation across a wide range of vision datasets.


## Key Features ✨

- 🔄 **Permutation Matching (Rebasin):** Compute and apply optimal permutations between models for effective model fusion, transfer, and task vector alignment.
- 🧭 **Task Vector Extraction & Transfer:** Extract task vectors from fine-tuned models and transfer them across different pre-training seeds or sources, even when models are not aligned.
- 🛠️ **Fine-tuning & Interpolation:** Utilities for fine-tuning and parameter interpolation.
- 📚 **Extensive Dataset Support:** Built-in loaders and templates for datasets such as CIFAR100, EuroSAT, SUN397, DTD, SVHN, GTSRB, RESISC45, ImageNet-R, Cars, and more.


## Installation ⚙️

Install the required dependencies:

```bash
pip install -r requirements.txt
# For additional modules, see also src/requirements.txt
```



## Usage 🧪


### Zero-shot Evaluation and Permutation Transfer ⚡

To perform zero-shot evaluation and permutation-based transfer between two CLIP models:

```bash
python main.py --arch <ARCH> \
    --pretraining_backbone_A <MODEL_A> \
    --pretraining_backbone_B <MODEL_B> \
    --finetuned_checkpoint_A <MODEL_A_FT>
    --dataset <DATASET> \
```

Replace the arguments with your desired configuration. See `utils.py` and `src/parser.py` for all available options.


### Fine-tuning 🔧

To fine-tune a CLIP model on a specific dataset:

```bash
python src/finetune_openCLIP.py --model_arch <ARCH> --dataset <DATASET> --num_steps <STEPS> --lr <LR> --batch_size <BATCH_SIZE> --wandb_project <WANDB_PROJECT>
```


### Example Datasets 🗂️

Supported datasets are implemented in `src/dataset/`. You can add new datasets by following the structure of existing dataset classes.



## Project Structure 📁

- `main.py` — Main script for permutation matching, task vector transfer, and evaluation.
- `src/` — Core modules: models, datasets, fine-tuning, interpolation.
- `permutations/` — Permutation specification and matching algorithms.
- `task_vectors/` — Task vector computation, transfer, and evaluation utilities.


## Contributing 🤝

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request.


## Citation 📚

If you use this code or ideas from the paper, please cite:

```
@misc{rinaldi2025updatetransformerlatestrelease,
      title={Update Your Transformer to the Latest Release: Re-Basin of Task Vectors}, 
      author={Filippo Rinaldi and Giacomo Capitani and Lorenzo Bonicelli and Donato Crisostomi and Federico Bolelli and Elisa Ficarra and Emanuele Rodolà and Simone Calderara and Angelo Porrello},
      year={2025},
      eprint={2505.22697},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.22697}, 
}
```

## Acknowledgments 🙏

This project builds on the work of:

- [OpenCLIP](https://github.com/mlfoundations/open_clip)  
- [Cycle-Consistent Model Merging](https://github.com/crisostomi/cycle-consistent-model-merging)  
- [Task Vectors](https://github.com/mlfoundations/task_vectors)  

We thank all contributors and the research community for their support.

For questions or issues, please open an issue on GitHub.