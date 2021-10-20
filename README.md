# efficient-task-transfer

This repository contains code for the experiments in our paper ["What to Pre-Train on? Efficient Intermediate Task Selection"](https://arxiv.org/pdf/2104.08247).
Most importantly, this includes scripts for easy training of **[Transformers](https://github.com/huggingface/transformers)** and **[Adapters](https://github.com/Adapter-Hub/adapter-transformers)** across [a wide range of NLU tasks](run_configs).

## Overview

The repository is structured as follows:
- **`itrain`** holds the `itrain` package which allows easy setup, training and evaluation of Transformers and Adapters
- **`run_configs`** provides default training configuration of all tasks currently supported by `itrain`
- **`training_scripts`** provides scripts for sequential adapter fine-tuning and adapter fusion as used in the paper
- **`task_selection`** provides scripts used for intermediate task selection in the paper

## Setup & Requirements

The code in this repository was developed using Python v3.6.8, PyTorch v1.7.1 and [adapter-transformers v1.1.1](https://github.com/Adapter-Hub/adapter-transformers), which is based on [HuggingFace Transformers v3.5.1](https://github.com/huggingface/transformers).
Using version different from the ones specified might not work.

After setting up Python and PyTorch (ideally in a virtual environment), all additional requirements together with the `itrain` package can be installed using:
```bash
pip install -e .
```

Additional setup steps required for running some scripts are detailed below locations.

## Transformer & Adapter Training

The `itrain` package provides a simple interface for configuring Transformer and Adapter training runs. `itrain` provides tools for:
- downloading and preprocessing datasets via HuggingFace datasets
- setting up Transformers and Adapter training
- training and evaluating on different tasks
- notifying on training start and results via mail or Telegram

`itrain` can be invoked from the command line by passing a run configuration file in json format.
Example configurations for all currently supported tasks can be found in the [run_configs](run_configs) folder.
All supported configuration keys are defined in [arguments.py](itrain/arguments.py).

Running a setup from the command line can look like this:
```bash
itrain --id 42 run_configs/sst2.json
```
This will train an adapter on the SST-2 task using `robert-base` as the base model (as specified in the config file).

Besides modifying configuration keys directly in the json file, they can be overriden using command line parameters.
E.g., we can modify the previous training run to fully fine-tune a `bert-base-uncased` model:
```bash
itrain --id <run_id> \
    --model_name_or_path bert-base-uncased \
    --train_adapter false \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --patience 0 \
    run_configs/<task>.json
```

Alternatively, training setups can be configured directly in Python by using the `Setup` class of `itrain`. An example for this is given in [example.py](example.py).

## Intermediate Task Transfer & Task Selection Experiments

Some scripts that helped running experiments presented in ["What to Pre-Train on? Efficient Intermediate Task Selection"](https://arxiv.org/pdf/2104.08247) are provided:
- See **[training_scripts](training_scripts)** for details on intermediate task transfer using sequential fine-tuning or adapter fusion
- See **[task_selection](task_selection)** for details on intermediate task selection methods.

All these scripts rely on pre-trained models/ adapters as described above and the following additional setup.

### Setup

We used a configuration file to specify the pre-trained models/ adapters and tasks to be used as transfer sources and transfer targets for different task transfer strategies and task selection methods.
The full configuration as used in the paper is given in [task_map.json](task_map.json).
It has to be modified to use self-trained models/ adapters:
- `from` and `to` specify which tasks are used as transfer source and transfer targets (names as defined in `run_configs`)
- `source_path_format` and `target_path_format` specify templates for the locations of pre-trained models/ adapters
- `adapters` provides a mapping from pre-trained (source) models/ adapters to run ids

Finally, the path to this task map and the folder holding the run configurations have to be made available to the scripts:

```bash
export RUN_CONFIG_DIR="/path/to/run_configs"
export DEFAULT_TASK_MAP="/path/to/task_map.json"
```

## Credits

- **[huggingface/transformers](https://github.com/huggingface/transformers)** for the Transformers implementations and the trainer class
- **[huggingface/datasets](https://github.com/huggingface/datasets)** for dataset downloading and preprocessing
- **[tuvuumass/task-transferability](https://github.com/tuvuumass/task-transferability)** for the TextEmb and TaskEmb implementations
- **[Adapter-Hub/adapter-transformers](https://github.com/Adapter-Hub/adapter-transformers)** for the adapter implementation

## Citation

If you find this repository helpful, please cite our paper ["What to Pre-Train on? Efficient Intermediate Task Selection"](https://arxiv.org/pdf/2104.08247):

```bibtex
@inproceedings{poth-etal-2021-what-to-pre-train-on,
    title={What to Pre-Train on? Efficient Intermediate Task Selection},
    author={Clifton Poth and Jonas Pfeiffer and Andreas Rücklé and Iryna Gurevych},
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/2104.08247",
    pages = "to appear",
}
```
