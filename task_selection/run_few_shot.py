"""
Runs few-shot sequential fine-tuning from source to target adapters.
"""

import argparse
import json
import logging
import os

import numpy as np
import torch
from run_embeddings import _restore_path
from run_model_features import compute_grad_features
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import AdamW, PreTrainedModel, get_linear_schedule_with_warmup
from utils import DEFAULT_TASK_MAP, MODEL_TYPE_MAP, RUN_CONFIG_DIR, get_full_name

from itrain import DATASET_MANAGER_CLASSES, DatasetArguments, DatasetManager, RunArguments
from itrain.arguments import ModelArguments
from itrain.model_creator import create_tokenizer
from itrain.runner import set_seed


logger = logging.getLogger(__name__)


def checkpoint_model(
    dataset_manager: DatasetManager,
    model: PreTrainedModel,
    run_args: RunArguments,
    checkpoint_steps: list,
    output_base,
):
    if not os.path.exists(output_base):
        os.makedirs(output_base)

    train_dataloader = DataLoader(
        dataset_manager.train_split,
        batch_size=run_args.batch_size,
        sampler=dataset_manager.train_sampler(),
        collate_fn=dataset_manager.collate_fn,
    )

    t_total = len(train_dataloader) // run_args.gradient_accumulation_steps * run_args.num_train_epochs

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

    logger.info("***** Running Training *****")
    logger.info("Num batches = %d", len(train_dataloader))
    logger.info("Batch size = %d", run_args.batch_size)

    model.zero_grad()
    train_iterator = trange(int(run_args.num_train_epochs), desc="Epoch", disable=False)
    global_step = 0
    for _ in train_iterator:
        num_examples = 0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        for step, inputs in enumerate(epoch_iterator):
            if run_args.max_steps > 0 and global_step >= run_args.max_steps:
                epoch_iterator.close()
                break

            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(run_args.device)

            outputs = model(**inputs)
            loss = outputs[0]
            loss.backward()

            if (step + 1) % run_args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), run_args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

            num_examples += inputs["input_ids"].size(0)

            if global_step in checkpoint_steps:
                checkpoint_dir = os.path.join(output_base, f"checkpoint-{global_step}")
                model.save_adapter(checkpoint_dir, model.active_adapters[0][0])


def run_probe_training(
    model_type: str,
    model_args: ModelArguments,
    adapter_dir: str,
    data_args: DatasetArguments,
    output_name: str,
    checkpoint_steps: list,
    output_base="output",
    overwrite=False,
    run_args=None,
    seed=42,
):
    output_dir = os.path.join(output_base, "checkpoints")
    run_args = run_args or RunArguments(
        batch_size=20,
        num_train_epochs=1,
    )
    full_output_name = get_full_name(output_name)

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger.warning("Device: %s", run_args.device)

    # Dataset
    dataset_manager: DatasetManager = DATASET_MANAGER_CLASSES[data_args.dataset_name](data_args)

    # Create output directory if needed
    model_output_dir = os.path.join(output_dir, dataset_manager.name, output_name)
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
    # skip computation if already exists
    elif not overwrite and all(
        [os.path.exists(os.path.join(model_output_dir, f"checkpoint-{step}")) for step in checkpoint_steps]
    ):
        logger.info("Output already exists, skipping {0}".format(model_output_dir))
        return

    # Set seed
    set_seed(seed)

    config_class, model_class = MODEL_TYPE_MAP[model_type]

    # Model config
    model_config = config_class.from_pretrained(
        model_args.model_name_or_path, **dataset_manager.get_model_config_kwargs()
    )
    tokenizer = create_tokenizer(model_args, **dataset_manager.get_tokenizer_config_kwargs())

    # Load dataset
    dataset_manager.tokenizer = tokenizer
    dataset_manager.load_and_preprocess()

    # Load model
    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        config=model_config,
    )
    model.load_adapter(adapter_dir, load_as=full_output_name, with_head=False)
    # add a simple prediction head if we require gradients
    head_config = dataset_manager.get_prediction_head_config()
    if "layers" in head_config:
        head_config["layers"] = 1
    model.add_prediction_head_from_config(full_output_name, head_config)
    model.set_active_adapters([full_output_name])
    model.train_adapter([full_output_name])

    model.to(run_args.device)

    checkpoint_model(dataset_manager, model, run_args, checkpoint_steps, model_output_dir)
    del model
    del dataset_manager


def run_probe_eval(
    model_type: str,
    model_args: ModelArguments,
    data_args: DatasetArguments,
    output_name: str,
    checkpoint_steps: list,
    checkpoint_base=None,
    output_base="output",
    overwrite=False,
    run_args=None,
    seed=42,
):
    if not checkpoint_base:
        checkpoint_base = os.path.join(output_base, "checkpoints")
    output_dir = os.path.join(output_base, "probe")
    run_args = run_args or RunArguments(
        batch_size=20,
        num_train_epochs=1,
    )
    full_output_name = get_full_name(output_name)

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger.warning("Device: %s", run_args.device)

    # Dataset
    dataset_manager: DatasetManager = DATASET_MANAGER_CLASSES[data_args.dataset_name](data_args)

    # Create output directory if needed
    model_output_dir = os.path.join(output_dir, dataset_manager.name, output_name)
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
    # skip computation if already exists
    elif os.path.exists(model_output_dir) and not overwrite:
        logger.info("Output already exists, skipping {0}".format(output_name))
        return

    # Set seed
    set_seed(seed)

    config_class, model_class = MODEL_TYPE_MAP[model_type]

    # Model config
    model_config = config_class.from_pretrained(
        model_args.model_name_or_path, retain_gradients=True, **dataset_manager.get_model_config_kwargs()
    )
    tokenizer = create_tokenizer(model_args, **dataset_manager.get_tokenizer_config_kwargs())
    dataset_manager.tokenizer = tokenizer

    # Load
    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        config=model_config,
    )

    dataset_manager.load_and_preprocess()
    model.to(run_args.device)

    # iterate over all checkpoints
    for checkpoint in checkpoint_steps:
        checkpoint_dir = os.path.join(checkpoint_base, dataset_manager.name, output_name, f"checkpoint-{checkpoint}")
        if not os.path.exists(checkpoint_dir):
            logger.info(f"Skipping {checkpoint_dir}")
            continue

        model.load_adapter(checkpoint_dir, load_as=full_output_name)
        model.set_active_adapters([full_output_name])
        model.train_adapter([full_output_name])
        model.to(run_args.device)

        run_args.max_steps = checkpoint
        feature_dict, metrics = compute_grad_features(
            dataset_manager, model, run_args, fisher=True, return_metrics=True
        )

        np.savez(os.path.join(model_output_dir, "features_{}.npz".format(checkpoint)), **feature_dict)
        with open(os.path.join(model_output_dir, "results_{}.json".format(checkpoint)), "w") as f:
            json.dump(metrics, f)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("mode", choices=["checkpoint", "probe"])
    parser.add_argument("--task_map", type=str, default=DEFAULT_TASK_MAP)
    parser.add_argument("--target_tasks", default=None, type=lambda s: s.split(","))
    parser.add_argument("--target_sizes", default="1000", type=lambda s: [int(item) for item in s.split(",")])
    parser.add_argument(
        "--checkpoint_steps", default="5,10,25,50", type=lambda s: [int(item) for item in s.split(",")]
    )
    parser.add_argument("-o", "--output_base", default="output", type=str)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
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
        run_args.batch_size = args.batch_size
        run_args.gradient_accumulation_steps = args.grad_accum_steps

        for train_size in args.target_sizes:
            data_args.train_subset_size = train_size

            # iterate source tasks
            for source_task in task_map["from"]:
                if args.mode == "checkpoint":
                    source_config_file = os.path.join(RUN_CONFIG_DIR, f"{source_task}.json")
                    with open(source_config_file, "r") as f:
                        source_config = json.load(f)
                    source_data_args = DatasetArguments(**source_config["dataset"])
                    adapter_dir = _restore_path(task_map, source_task, source_data_args)

                    run_probe_training(
                        args.model_type,
                        model_args,
                        adapter_dir,
                        data_args,
                        source_task,
                        args.checkpoint_steps,
                        output_base=args.output_base,
                        overwrite=args.overwrite,
                        run_args=run_args,
                    )
                elif args.mode == "probe":
                    run_probe_eval(
                        args.model_type,
                        model_args,
                        data_args,
                        source_task,
                        args.checkpoint_steps,
                        output_base=args.output_base,
                        overwrite=args.overwrite,
                        run_args=run_args,
                    )
                else:
                    raise ValueError(f"Invalid mode {args.mode}.")

                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
