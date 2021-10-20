"""
Compute the encoder features of a dataset on a set of source models.
"""

import argparse
import json
import logging
import os

import numpy as np
import torch
from run_embeddings import _restore_path
from taskemb import TaskEmbArguments, compute_Fisher
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import AdamW, PreTrainedModel, get_linear_schedule_with_warmup
from transformers.trainer_pt_utils import nested_numpify
from utils import DEFAULT_TASK_MAP, MODEL_TYPE_MAP, RUN_CONFIG_DIR

from itrain import DATASET_MANAGER_CLASSES, DatasetArguments, DatasetManager, RunArguments
from itrain.arguments import ModelArguments
from itrain.datasets.tagging import TaggingDatasetManager
from itrain.model_creator import create_tokenizer
from itrain.runner import _pad_and_concatenate, set_seed


logger = logging.getLogger(__name__)


def compute_output_features(dataset_manager: DatasetManager, model: PreTrainedModel, run_args: RunArguments):
    train_dataloader = DataLoader(
        dataset_manager.train_split,
        batch_size=run_args.batch_size,
        sampler=dataset_manager.train_sampler(),
        collate_fn=dataset_manager.collate_fn,
    )

    logger.info("***** Compute Model Output Features *****")
    logger.info("Num batches = %d", len(train_dataloader))
    logger.info("Batch size = %d", run_args.batch_size)

    model.eval()
    model.zero_grad()

    num_examples = 0
    output_features = None
    labels = None

    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)

    for step, inputs in enumerate(epoch_iterator):
        do_averaging = True
        # don't input labels as we don't have a head
        if "labels" in inputs:
            batch_labels = inputs.pop("labels")
            do_averaging = len(batch_labels.shape) <= 1
            batch_labels = batch_labels.reshape(-1, 1)
        elif "start_positions" in inputs:
            do_averaging = False
            batch_labels = 1 * inputs.pop("start_positions") + 2 * inputs.pop("end_positions")
            batch_labels = batch_labels.reshape(-1, 1)
        if labels is None:
            labels = batch_labels
        else:
            labels = np.vstack((labels, batch_labels))

        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(run_args.device)

        with torch.no_grad():
            input_mask = inputs["attention_mask"]
            outputs = model(**inputs)
            sequence_output = outputs[0]  # batch_size x max_seq_length x hidden_size

            input_mask = input_mask.view(-1, input_mask.size(-1))
            active_sequence_output = torch.einsum("ijk,ij->ijk", [sequence_output, input_mask])
            # batch_size x hidden_size
            if do_averaging:
                avg_sequence_output = active_sequence_output.sum(1) / input_mask.sum(dim=1).view(input_mask.size(0), 1)
            else:
                # special handling of tagging tasks & QA tasks
                avg_sequence_output = active_sequence_output.view(batch_labels.shape[0], -1)
            avg_sequence_output = avg_sequence_output.detach().cpu().numpy()

            # special handling of multiple-choice tasks
            if avg_sequence_output.shape[0] != batch_labels.shape[0]:
                avg_sequence_output = avg_sequence_output.reshape(
                    batch_labels.shape[0], -1, avg_sequence_output.shape[-1]
                )

            if output_features is None:
                output_features = avg_sequence_output
            else:
                output_features = np.vstack((output_features, avg_sequence_output))

        num_examples += batch_labels.shape[0]

    assert output_features.shape[0] == num_examples
    assert labels.shape[0] == num_examples

    return output_features, labels


def _get_gradients(model):
    outputs = {}

    base_model = model.base_model
    for name, parameter in base_model.named_parameters():
        if parameter.requires_grad:
            score = parameter.grad
            if score is not None and name not in outputs:
                outputs[name] = score

    return outputs


def compute_grad_features(
    dataset_manager: DatasetManager,
    model: PreTrainedModel,
    run_args: RunArguments,
    finetune=True,
    fisher=False,
    return_metrics=False,
):
    train_dataloader = DataLoader(
        dataset_manager.train_split,
        batch_size=run_args.batch_size,
        sampler=dataset_manager.train_sampler(),
        collate_fn=dataset_manager.collate_fn,
    )

    t_total = len(train_dataloader) // run_args.gradient_accumulation_steps * run_args.num_train_epochs

    if finetune:
        model.train()

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": run_args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=run_args.learning_rate, eps=run_args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=run_args.warmup_steps, num_training_steps=t_total
        )

    else:
        model.eval()
        optimizer = None
        scheduler = None

    logger.info("***** Compute Model Grad Features *****")
    logger.info("Num batches = %d", len(train_dataloader))
    logger.info("Batch size = %d", run_args.batch_size)
    total_num_examples = 0
    global_step = 0
    preds = None
    label_ids = None

    model.zero_grad()
    train_iterator = trange(int(run_args.num_train_epochs), desc="Epoch", disable=False)
    global_feature_dict = {}
    for _ in train_iterator:
        num_examples = 0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        for _, inputs in enumerate(epoch_iterator):
            if global_step >= run_args.max_steps:
                epoch_iterator.close()
                break

            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(run_args.device)

            outputs = model(**inputs)
            loss = outputs[0]
            model.zero_grad()
            loss.backward()

            if fisher:
                taskemb_args = TaskEmbArguments()
                input_mask = inputs["attention_mask"]
                input_mask = input_mask.view(-1, input_mask.size(-1))
                total_tokens = input_mask.float().detach().sum().data
                feature_dict = compute_Fisher(taskemb_args, model, input_mask, total_tokens)
            else:
                feature_dict = _get_gradients(model)

            if len(global_feature_dict) == 0:
                for key in feature_dict:
                    global_feature_dict[key] = feature_dict[key].detach().cpu().numpy()
            else:
                for key in feature_dict:
                    global_feature_dict[key] += feature_dict[key].detach().cpu().numpy()

            if return_metrics:
                logits = tuple(logit.detach() for logit in outputs[1:])
                if len(logits) == 1:
                    logits = logits[0]
                labels = tuple(inputs.get(name).detach() for name in dataset_manager.label_column_names)
                if len(labels) == 1:
                    labels = labels[0]

                if logits is not None:
                    preds = (
                        nested_numpify(logits)
                        if preds is None
                        else _pad_and_concatenate(preds, nested_numpify(logits), padding_index=-100)
                    )
                if labels is not None:
                    label_ids = (
                        nested_numpify(labels)
                        if label_ids is None
                        else _pad_and_concatenate(label_ids, nested_numpify(labels), padding_index=-100)
                    )

            if finetune:
                torch.nn.utils.clip_grad_norm_(model.parameters(), run_args.max_grad_norm)
                optimizer.step()
                scheduler.step()

            model.zero_grad()
            num_examples += inputs["input_ids"].size(0)
            global_step += 1

        total_num_examples += num_examples

    # Normalize
    for key in global_feature_dict:
        global_feature_dict[key] = global_feature_dict[key] / total_num_examples

    if return_metrics:
        metrics = dataset_manager.compute_metrics(preds, label_ids)
        return global_feature_dict, metrics

    return global_feature_dict


def run_model_features(
    model_type: str,
    model_args: ModelArguments,
    adapter_dir: str,
    data_args: DatasetArguments,
    output_name: str,
    gradients=False,
    output_base="output",
    overwrite=False,
    run_args=None,
    fisher=False,
    seed=42,
):
    if gradients and fisher:
        output_dir = os.path.join(output_base, "model_fisher_features")
    elif gradients:
        output_dir = os.path.join(output_base, "model_grad_features")
    else:
        output_dir = os.path.join(output_base, "model_output_features")
    run_args = run_args or RunArguments(
        batch_size=32,
        num_train_epochs=1,
    )

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger.warning("Device: %s", run_args.device)

    # Dataset
    dataset_manager: DatasetManager = DATASET_MANAGER_CLASSES[data_args.dataset_name](data_args)

    # Create output directory if needed
    task_output_dir = os.path.join(output_dir, dataset_manager.name)
    if not os.path.exists(task_output_dir):
        os.makedirs(task_output_dir)
    # skip computation if already exists
    elif os.path.exists(os.path.join(task_output_dir, "{}.npz".format(output_name))) and not overwrite:
        logger.info("Output already exists, skipping {0}".format(output_name))
        return

    # Set seed
    set_seed(seed)

    config_class, model_class = MODEL_TYPE_MAP[model_type]

    # Model config
    model_config = config_class.from_pretrained(
        model_args.model_name_or_path,
        retain_gradients=gradients,
    )
    if isinstance(dataset_manager, TaggingDatasetManager):
        add_prefix_space = True
    else:
        add_prefix_space = False
    tokenizer = create_tokenizer(model_args, add_prefix_space=add_prefix_space)
    dataset_manager.tokenizer = tokenizer

    # Load
    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        config=model_config,
    )
    adapter_name = model.load_adapter(adapter_dir, with_head=False)
    # add a simple prediction head if we require gradients
    if gradients:
        head_config = dataset_manager.get_prediction_head_config()
        head_config["layers"] = 1
        model.add_prediction_head_from_config(adapter_name, head_config)
    model.set_active_adapters([adapter_name])
    model.train_adapter([adapter_name])

    dataset_manager.load_and_preprocess()
    model.to(run_args.device)

    if gradients:
        feature_dict = compute_grad_features(dataset_manager, model, run_args, fisher=fisher)

        np.savez(os.path.join(task_output_dir, "{}.npz".format(output_name)), **feature_dict)
    else:
        features, labels = compute_output_features(dataset_manager, model, run_args)

        np.savez(os.path.join(task_output_dir, "{}.npz".format(output_name)), features=features, labels=labels)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task_map", type=str, default=DEFAULT_TASK_MAP)
    parser.add_argument("--target_tasks", default=None, type=lambda s: s.split(","))
    parser.add_argument("--target_sizes", default="1000", type=lambda s: [int(item) for item in s.split(",")])
    parser.add_argument("-o", "--output_base", default="output", type=str)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--gradients", action="store_true")
    parser.add_argument("--fisher", action="store_true")
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--model_type", default="roberta")
    parser.add_argument("--model_name_or_path", default=None)

    args = parser.parse_args()

    with open(os.path.expanduser(args.task_map), "r") as f:
        task_map = json.load(f)
    target_tasks = args.target_tasks or task_map["to"].keys()

    for target_task in target_tasks:
        # target task config
        target_config_file = os.path.join(RUN_CONFIG_DIR, f"{target_task}.json")
        with open(target_config_file, "r") as f:
            target_config = json.load(f)
        data_args = DatasetArguments(**target_config["dataset"])
        model_args = ModelArguments(**target_config["model"])
        if args.model_name_or_path:
            model_args.model_name_or_path = args.model_name_or_path
        run_args = RunArguments(**target_config["training"])
        run_args.num_train_epochs = 1
        if args.max_steps > 0:
            run_args.max_steps = args.max_steps
        if args.batch_size is not None:
            run_args.batch_size = args.batch_size

        for train_size in args.target_sizes:
            data_args.train_subset_size = train_size

            # iterate source tasks
            for source_task in task_map["from"]:
                source_config_file = os.path.join(RUN_CONFIG_DIR, f"{source_task}.json")
                with open(source_config_file, "r") as f:
                    source_config = json.load(f)
                source_data_args = DatasetArguments(**source_config["dataset"])
                adapter_dir = _restore_path(task_map, source_task, source_data_args)

                run_model_features(
                    args.model_type,
                    model_args,
                    adapter_dir,
                    data_args,
                    source_task,
                    gradients=args.gradients,
                    output_base=args.output_base,
                    overwrite=args.overwrite,
                    run_args=run_args,
                    fisher=args.fisher,
                )


if __name__ == "__main__":
    main()
