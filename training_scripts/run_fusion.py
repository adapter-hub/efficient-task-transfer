import argparse
import json
import os

from config_utils import DEFAULT_TASK_MAP, get_dataset_config, restore_path

from itrain import ModelArguments, RunArguments, Setup


FUSION_OUTPUT_DIR = "fusion_output"
FUSION_FINETUNE_OUTPUT_DIR = "fusion_finetune_output"


def run_fusion(args, config_name=None):
    # init setup
    dataset_manager, config = get_dataset_config(args["target_task"], train_size=args["train_size"])
    if args["train_size"] > 0:
        target_task_name = args["target_task"] + "_n" + str(args["train_size"])
    else:
        target_task_name = args["target_task"]
    if args["finetune_adapters"]:
        output_base = os.path.join(FUSION_FINETUNE_OUTPUT_DIR, "to_" + target_task_name)
    else:
        output_base = os.path.join(FUSION_OUTPUT_DIR, "to_" + target_task_name)
    # patch training args
    config["training"]["learning_rate"] = args["learning_rate"]
    config["training"]["num_train_epochs"] = args["num_train_epochs"]

    # load results if existing
    final_results_file = os.path.join(output_base, "eval_results.json")
    if os.path.exists(final_results_file):
        with open(final_results_file, "r") as f:
            results = json.load(f)
    else:
        results = {}

    with open(os.path.expanduser(args["task_map"]), "r") as f:
        task_map = json.load(f)

    fusion_name = ",".join(args["source_tasks"])
    print(f"*** Running fusion from {fusion_name} to {target_task_name} ***")
    output_dir = os.path.join(output_base, fusion_name)
    # skip this iteration if no overwrites requested & existing
    if args["overwrite_mode"] == 0 and os.path.exists(output_dir):
        print(f"Skipping task {fusion_name} as it already exists.")
        return

    # setup the dataset and training params
    setup = Setup(id=args["id"])
    setup.dataset(dataset_manager)
    config["training"]["output_dir"] = output_dir
    setup.training(RunArguments(**config["training"]))
    if isinstance(config["evaluation"], str):
        setup.evaluation(split=config["evaluation"])
    else:
        setup.evaluation()
    setup.notify(config["notify"])
    setup._config_name = config_name or "fusion_" + fusion_name + "_to_" + target_task_name

    # setup model
    load_adapters_map = {}
    # iterate over all adapters for fusion
    for source_task in args["source_tasks"]:
        source_dataset_manager, _ = get_dataset_config(source_task)
        load_adapters_map[source_task] = restore_path(task_map, source_task, source_dataset_manager)
    # patch model args
    config["model"]["load_adapters"] = load_adapters_map
    config["model"]["train_adapter_fusion"] = fusion_name
    config["model"]["drop_last_fusion_layer"] = True
    config["model"]["train_adapter"] = args["finetune_adapters"]
    setup.model(ModelArguments(**config["model"]))
    # start!
    if fusion_name in results and args["overwrite_mode"] == 1:
        # append to existing
        run_results = setup.run(restarts=args["restarts"], first_run_index=len(results[fusion_name]["seeds"]))
        for k, v in run_results.items():
            results[fusion_name][k] += v
    else:
        run_results = setup.run(restarts=args["restarts"])
        results[fusion_name] = run_results

    # save results
    with open(final_results_file, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("target_task", type=str, help="Name of the target task training setup.")
    parser.add_argument("--id", type=int, default=0, help="ID of this run.")
    parser.add_argument("--task_map", type=str, default=DEFAULT_TASK_MAP)
    parser.add_argument(
        "--overwrite_mode", type=int, choices=[0, 1, 2], default=0, help="0: no overwrite; 1: append; 2: overwrite"
    )
    parser.add_argument("--source_tasks", type=lambda s: s.split(","), required=True)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=int, default=15)
    parser.add_argument("--train_size", type=int, default=-1)
    parser.add_argument("--restarts", type=int, default=None)
    parser.add_argument("--finetune_adapters", action="store_true")
    args = vars(parser.parse_args())

    run_fusion(args)
