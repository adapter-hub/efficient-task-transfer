"""
Compute SEmb for classification/ regression/ question-answering tasks.
"""

import argparse
import logging
import os

import numpy as np
import torch
from itrain import DATASET_MANAGER_CLASSES, DatasetArguments, DatasetManager, RunArguments
from itrain.datasets.tagging import TaggingDatasetManager
from itrain.runner import set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import AutoModel, AutoTokenizer, PreTrainedModel


logger = logging.getLogger(__name__)


def compute_semb(dataset_manager: DatasetManager, model: PreTrainedModel, run_args: RunArguments):
    train_dataloader = DataLoader(
        dataset_manager.train_split,
        batch_size=run_args.batch_size,
        sampler=dataset_manager.train_sampler(),
        collate_fn=dataset_manager.collate_fn,
    )

    logger.info("***** Compute SEmb *****")
    logger.info("Num batches = %d", len(train_dataloader))
    logger.info("Batch size = %d", run_args.batch_size)

    model.eval()
    model.zero_grad()
    train_iterator = trange(int(run_args.num_train_epochs), desc="Epoch", disable=False)

    total_num_examples = 0
    global_feature_dict = {}
    for _ in train_iterator:
        num_examples = 0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)

        for step, inputs in enumerate(epoch_iterator):
            # don't input labels as we don't have a head
            if "labels" in inputs:
                del inputs["labels"]
            elif "start_positions" in inputs:
                del inputs["start_positions"]
                del inputs["end_positions"]
            # HACK
            if model.config.model_type == "distilbert" and "token_type_ids" in inputs:
                del inputs["token_type_ids"]
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(run_args.device).view(-1, v.size(-1))

            with torch.no_grad():
                input_mask = inputs["attention_mask"]
                outputs = model(**inputs)
                sequence_output = outputs[0]  # batch_size x max_seq_length x hidden_size

                input_mask = input_mask.view(-1, input_mask.size(-1))
                active_sequence_output = torch.einsum("ijk,ij->ijk", [sequence_output, input_mask])
                avg_sequence_output = active_sequence_output.sum(1) / input_mask.sum(dim=1).view(input_mask.size(0), 1)

                if len(global_feature_dict) == 0:
                    global_feature_dict["avg_sequence_output"] = avg_sequence_output.sum(dim=0).detach().cpu().numpy()
                else:
                    global_feature_dict["avg_sequence_output"] += avg_sequence_output.sum(dim=0).detach().cpu().numpy()

            num_examples += input_mask.size(0)
        total_num_examples += num_examples

    # Normalize
    for key in global_feature_dict:
        global_feature_dict[key] = global_feature_dict[key] / total_num_examples

    return global_feature_dict


def run_semb(args, data_args=None, run_args=None, seed=42):
    run_args = run_args or RunArguments(
        batch_size=32,
        num_train_epochs=1,
    )

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger.warning("Device: %s", run_args.device)

    # Dataset
    data_args = data_args or DatasetArguments(dataset_name=args["dataset"], task_name=args["dataset_task"])
    dataset_manager: DatasetManager = DATASET_MANAGER_CLASSES[data_args.dataset_name](data_args)

    # Create output directory if needed
    task_output_dir = os.path.join(args["output_dir"], dataset_manager.name)
    if not os.path.exists(task_output_dir):
        os.makedirs(task_output_dir)
    # skip computation if already exists
    elif len(os.listdir(task_output_dir)) > 0 and not args["overwrite"]:
        logger.info("Output already exists, skipping {0}".format(dataset_manager.name))
        return

    # Set seed
    set_seed(seed)

    # Tokenizer
    if isinstance(dataset_manager, TaggingDatasetManager):
        add_prefix_space = True
    else:
        add_prefix_space = False
    tokenizer = AutoTokenizer.from_pretrained(
        args["model_name"],
        use_fast=args.get("fast_tokenizer", False),
        add_prefix_space=add_prefix_space,
    )
    dataset_manager.tokenizer = tokenizer

    # Load
    model = AutoModel.from_pretrained(args["model_name"])
    dataset_manager.load_and_preprocess()
    model.to(run_args.device)

    feature_dict = compute_semb(dataset_manager, model, run_args)

    for key in feature_dict:
        np.save(os.path.join(task_output_dir, "{}.npy".format(key)), feature_dict[key])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default=None, type=str, required=True)
    parser.add_argument("--dataset_task", default=None, type=str)
    parser.add_argument("--model_name", default="sentence-transformers/roberta-base-nli-stsb-mean-tokens", type=str)
    parser.add_argument("--output_dir", default="output/semb", type=str)
    parser.add_argument("--overwrite", action="store_true", default=False)

    args = parser.parse_args()

    run_semb(vars(args))


if __name__ == "__main__":
    main()
