#!/usr/bin/env python3
import argparse
import json
import logging
import os

from semb import run_semb
from taskemb import run_taskemb
from textemb import run_textemb
from utils import DEFAULT_TASK_MAP, RUN_CONFIG_DIR

from itrain import DatasetArguments, RunArguments


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _restore_path(adapter_map, task_name, dataset_args):
    template = adapter_map["source_path_format"]
    run_id = adapter_map["adapters"][task_name]
    # get the actual manager name
    if dataset_args.task_name:
        manager_name = f"{dataset_args.dataset_name}_{dataset_args.task_name}"
    else:
        manager_name = dataset_args.dataset_name
    # HACK: the actual path to the adapter may have different names
    path = os.path.expanduser(template.format(task_name, run_id, manager_name))
    if not os.path.exists(path):
        path = os.path.expanduser(template.format(manager_name, run_id, manager_name))
    if not os.path.exists(path):
        path = os.path.expanduser(template.format(task_name, run_id, task_name))
    return path


def _restore_target_path(adapter_map, task_name, dataset_args):
    template = adapter_map["target_path_format"]
    # get the actual manager name
    if dataset_args.task_name:
        manager_name = f"{dataset_args.dataset_name}_{dataset_args.task_name}_n{dataset_args.train_subset_size}"
    else:
        manager_name = f"{dataset_args.dataset_name}_n{dataset_args.train_subset_size}"
    # HACK: the actual path to the adapter may have different names
    path = os.path.expanduser(template.format(task_name, dataset_args.train_subset_size, manager_name))
    if not os.path.exists(path):
        path = os.path.expanduser(template.format(manager_name, dataset_args.train_subset_size, manager_name))
    if not os.path.exists(path):
        path = os.path.expanduser(template.format(task_name, dataset_args.train_subset_size, task_name))
    return path


def run(args, load_dir, data_args, run_config, overwrite=False, seed=42):
    logger.info(f"***** Compute measures for {load_dir} *****\n")
    output_base = args.output_dir or ("full_output" if args.full_model else "output")

    if args.textemb:
        run_textemb(
            {
                "model_name": args.model_name or "roberta-base",
                "fast_tokenizer": run_config["model"].get("use_fast_tokenizer", False),
                "output_dir": os.path.join(output_base, "textemb"),
                "overwrite": overwrite,
            },
            data_args=data_args,
            seed=seed,
        )
    if args.semb:
        model_name = args.model_name or "sentence-transformers/roberta-base-nli-stsb-mean-tokens"
        run_semb(
            {
                "model_name": model_name,
                "fast_tokenizer": run_config["model"].get("use_fast_tokenizer", False),
                "output_dir": os.path.join(output_base, "semb", model_name.split("/")[-1]),
                "overwrite": overwrite,
            },
            data_args=data_args,
            seed=seed,
        )
    if args.taskemb:
        run_args = RunArguments(**run_config["training"])
        run_args.num_train_epochs = 1
        task_args = {
            "model_type": args.model_type,
            "model_name": args.model_name or "roberta-base",
            "fast_tokenizer": run_config["model"].get("use_fast_tokenizer", False),
            "output_dir": os.path.join(output_base, "taskemb"),
            "overwrite": overwrite,
        }
        if args.full_model:
            task_args["model_dir"] = load_dir
        else:
            task_args["adapter_dir"] = load_dir
        run_taskemb(
            task_args,
            data_args=data_args,
            run_args=run_args,
            seed=seed,
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task_map", type=str, default=DEFAULT_TASK_MAP)
    parser.add_argument("--taskemb", action="store_true")
    parser.add_argument("--textemb", action="store_true")
    parser.add_argument("--semb", action="store_true")
    parser.add_argument("-o", "--output_dir", default=None, type=str)
    parser.add_argument("--overwrite", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--target_tasks", action="store_true")
    group.add_argument("--source_tasks", action="store_true")
    group.add_argument("--tasks", default="", type=lambda s: s.split(","))
    parser.add_argument("--dataset_sizes", default="0", type=lambda s: [int(item) for item in s.split(",")])
    parser.add_argument("--model_type", default="roberta", type=str)
    parser.add_argument("--model_name", default=None, type=str)
    parser.add_argument("--full_model", action="store_true")
    parser.add_argument("--seed", default=42, type=int)

    args = parser.parse_args()

    with open(os.path.expanduser(args.task_map), "r") as f:
        task_map = json.load(f)

    # source tasks
    if args.source_tasks:
        for dataset in task_map["from"]:
            dataset_cleaned = dataset.replace("-", "")
            run_config_file = os.path.join(RUN_CONFIG_DIR, f"{dataset_cleaned}.json")
            with open(run_config_file, "r") as f:
                run_config = json.load(f)
            for train_size in args.dataset_sizes:
                run_config["dataset"]["train_subset_size"] = train_size
                data_args = DatasetArguments(**run_config["dataset"])
                adapter_dir = _restore_path(task_map, dataset, data_args)

                run(args, adapter_dir, data_args, run_config, overwrite=args.overwrite, seed=args.seed)

    # user-specified tasks
    if args.tasks:
        for task in args.tasks:
            if not task:
                continue
            run_config_file = os.path.join(RUN_CONFIG_DIR, f"{task}.json")
            with open(run_config_file, "r") as f:
                run_config = json.load(f)
            for train_size in args.dataset_sizes:
                run_config["dataset"]["train_subset_size"] = train_size
                data_args = DatasetArguments(**run_config["dataset"])
                adapter_dir = _restore_path(task_map, task, data_args)

                run(args, adapter_dir, data_args, run_config, overwrite=args.overwrite, seed=args.seed)

    # target tasks
    if args.target_tasks:
        for dataset in task_map["to"]:
            run_config_file = os.path.join(RUN_CONFIG_DIR, f"{dataset}.json")
            with open(run_config_file, "r") as f:
                run_config = json.load(f)
            for train_size in args.dataset_sizes:
                run_config["dataset"]["train_subset_size"] = train_size
                data_args = DatasetArguments(**run_config["dataset"])
                adapter_dir = _restore_target_path(task_map, dataset, data_args)

                run(args, adapter_dir, data_args, run_config, overwrite=args.overwrite, seed=args.seed)


if __name__ == "__main__":
    main()
