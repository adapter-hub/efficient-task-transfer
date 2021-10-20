import json
import os

from modeling.configuration_bert import BertConfig
from modeling.configuration_roberta import RobertaConfig
from modeling.modeling_bert import BertModelWithHeads
from modeling.modeling_roberta import RobertaModelWithHeads


RUN_CONFIG_DIR = os.path.expanduser(os.getenv("RUN_CONFIG_DIR", "../run_configs"))
DEFAULT_TASK_MAP = os.path.expanduser(os.getenv("DEFAULT_TASK_MAP", "~/itrain/trained_adapter_map.json"))

MODEL_TYPE_MAP = {
    "bert": (BertConfig, BertModelWithHeads),
    "roberta": (RobertaConfig, RobertaModelWithHeads),
}


cache = {}


def get_full_name(short_name, size=None):
    short_name = short_name.replace("-", "")
    name = short_name + "_n" + str(size)
    if name not in cache:
        config_file = os.path.join(RUN_CONFIG_DIR, f"{short_name}.json")
        with open(config_file, "r") as f:
            config = json.load(f)
        if "task_name" in config["dataset"]:
            cache[name] = f"{config['dataset']['dataset_name']}_{config['dataset']['task_name']}"
        else:
            cache[name] = config["dataset"]["dataset_name"]
        if size is not None:
            cache[name] += f"_n{size}"
    return cache[name]
