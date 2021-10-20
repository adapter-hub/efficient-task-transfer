import json
import os

from itrain import DATASET_MANAGER_CLASSES, DatasetArguments


RUN_CONFIGS = os.path.expanduser(os.getenv("RUN_CONFIG_DIR", "~/itrain/run_configs"))
DEFAULT_TASK_MAP = os.path.expanduser(os.getenv("DEFAULT_TASK_MAP", "~/itrain/trained_adapter_map.json"))


def get_dataset_config(config_name, train_size=-1):
    # init setup
    with open(os.path.join(RUN_CONFIGS, config_name + ".json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    # dataset manager
    config["dataset"]["train_subset_size"] = train_size
    dataset_args = DatasetArguments(**config["dataset"])
    dataset_manager = DATASET_MANAGER_CLASSES[dataset_args.dataset_name](dataset_args)
    return dataset_manager, config


def restore_path(adapter_map, task_name, manager):
    template = adapter_map["source_path_format"]
    run_id = adapter_map["adapters"][task_name]
    # HACK: the actual path to the adapter may have different names
    path = os.path.expanduser(template.format(task_name, run_id, manager.name))
    if not os.path.exists(path):
        path = os.path.expanduser(template.format(manager.name, run_id, manager.name))
    return path
