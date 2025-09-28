# [WWW 2026] Inductive Graph on Dynamic Graph: An Inductive Approach for Capturing Global Dynamic Evolutionary Information in Dynamic Graphs(IGDG)

------

## 1. Requirements

Main package requirements:

- `CUDA >= 10.1`
- `Python >= 3.8.0`
- `PyTorch >= 1.9.1`
- `PyTorch-Geometric >= 2.0.1`

To install the complete requiring packages, use the following command at the root directory of the repository:

```setup
pip install -r requirements.txt
```

## 2. Quick Start

### Training

To train IGDG, run the following command in the directory `./scripts`:

```train
python main.py --dataset=<data_name> --device_id=<gpu>  
```

## 4. Acknowledgements

Part of this code is inspired by Tailin Wu et al.'s [GIB](https://github.com/snap-stanford/GIB) . We owe sincere thanks to their valuable efforts and contributions.
