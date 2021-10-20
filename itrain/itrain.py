import dataclasses
import json
import logging
import os
from collections import defaultdict
from typing import Optional, Sequence, Union

import numpy as np
import transformers
from transformers import PreTrainedModel

from .arguments import DatasetArguments, ModelArguments, RunArguments
from .datasets import DATASET_MANAGER_CLASSES, CacheMode, DatasetManager
from .model_creator import create_model, create_tokenizer, register_heads
from .notifier import NOTIFIER_CLASSES
from .trainer import Trainer, set_seed


OUTPUT_FOLDER = os.environ.get("ITRAIN_OUTPUT") or "run_output"
LOGS_FOLDER = os.environ.get("ITRAIN_LOGS") or "run_logs"

logging.basicConfig(level=logging.INFO)
transformers.logging.set_verbosity_info()
logger = logging.getLogger(__name__)


class Setup:
    """
    Defines, configures and runs a full Transformer training setup,
    including dataset configuration, model setup, training and evaluation.
    """

    id: int
    dataset_manager: DatasetManager
    model_instance: PreTrainedModel
    model_args: ModelArguments

    def __init__(self, id=0):
        self.id = id
        self.dataset_manager = None
        self.model_instance = None
        self.model_args = None
        self._train_run_args = None
        self._do_eval = False
        self._eval_run_args = None
        self._eval_split = None
        self._notifiers = {}
        self._config_name = None

    @property
    def name(self):
        return self._config_name or self.dataset_manager.name

    def dataset(self, args_or_manager: Union[DatasetArguments, DatasetManager]):
        """
        Set up the dataset.
        """
        if self.dataset_manager is not None:
            raise ValueError("Dataset already set.")
        if isinstance(args_or_manager, DatasetManager):
            self.dataset_manager = args_or_manager
        else:
            self.dataset_manager = DATASET_MANAGER_CLASSES[args_or_manager.dataset_name](args_or_manager)

    def model(self, args: ModelArguments):
        """
        Set up the model.
        """
        if self.model_instance is not None:
            raise ValueError("Model already set.")
        self.model_args = args

    def training(self, args: RunArguments):
        """
        Set up training.
        """
        self._train_run_args = args

    def evaluation(self, args: Optional[RunArguments] = None, split=None):
        """
        Set up evaluation.
        """
        self._do_eval = True
        self._eval_run_args = args
        self._eval_split = split

    def notify(self, notifier_name: str, **kwargs):
        """
        Set up a notifier. Can be either "telegram" or "email" currently.
        """
        self._notifiers[notifier_name] = NOTIFIER_CLASSES[notifier_name](**kwargs)

    def _override(self, instance, overrides):
        orig_dict = dataclasses.asdict(instance)
        for k in orig_dict.keys():
            if k in overrides:
                v = overrides.pop(k)
                if v is not None:
                    orig_dict[k] = v
        return type(instance)(**orig_dict)

    def load_from_file(self, file, overrides=None):
        """
        Load a setup from file.
        """
        with open(file, "r", encoding="utf-8") as f:
            config = json.load(f)
        dataset_args = DatasetArguments(**config["dataset"])
        if overrides:
            dataset_args = self._override(dataset_args, overrides)
        self.dataset(dataset_args)
        model_args = ModelArguments(**config["model"])
        if overrides:
            model_args = self._override(model_args, overrides)
        self.model(model_args)
        if "training" in config:
            train_args = RunArguments(**config["training"])
            if overrides:
                train_args = self._override(train_args, overrides)
            self.training(train_args)
        if "evaluation" in config:
            if isinstance(config["evaluation"], dict):
                eval_args = RunArguments(**config["evaluation"])
                if overrides:
                    eval_args = self._override(eval_args, overrides)
                self.evaluation(eval_args)
            elif isinstance(config["evaluation"], str):
                self.evaluation(split=config["evaluation"])
            elif config["evaluation"]:
                self.evaluation()
        if "notify" in config:
            self.notify(config["notify"])
        self._config_name = os.path.splitext(os.path.basename(file))[0]

    def _setup_tokenizer(self):
        self.dataset_manager.tokenizer = create_tokenizer(
            self.model_args,
            **self.dataset_manager.get_tokenizer_config_kwargs(),
        )

    def _prepare_run_args(self, args: RunArguments, restart=None):
        if not args.output_dir:
            args.output_dir = os.path.join(
                OUTPUT_FOLDER, self.model_instance.config.model_type, self.name, str(self.id)
            )
        if not args.logging_dir:
            args.logging_dir = os.path.join(
                LOGS_FOLDER, "_".join([str(self.id), self.model_instance.config.model_type, self.name])
            )
        # create a copy with the run-specific adjustments
        prepared_args = RunArguments(**dataclasses.asdict(args))
        if restart is not None:
            prepared_args.output_dir = os.path.join(args.output_dir, str(restart))
            prepared_args.logging_dir = os.path.join(args.logging_dir, str(restart))
        return prepared_args

    def run(self, restarts=None, first_run_index=0):
        """
        Run this setup. Dataset, model, and training or evaluation are expected to be set.

        Args:
            restarts (Union[list, int], optional): Defines the random training restarts. Can be either:
                - a list of integers: run training with each of the given values as random seed.
                - a single integer: number of random restarts, each with a random seed.
                - None (default): one random restart.
            first_run_index (int, optional): The start index in the sequence of random restarts. Defaults to 0.

        Returns:
            dict: A dictionary of run results containing the used random seeds and (optionally) evaluation results.
        """
        # Set up tokenizer
        self._setup_tokenizer()
        # Load dataset
        self.dataset_manager.load_and_preprocess()

        if not restarts:
            restarts = [42]
        if not isinstance(restarts, Sequence):
            restarts = [None for _ in range(int(restarts))]
        has_restarts = len(restarts) > 1
        # collect results
        all_results = defaultdict(list)

        # Init notifier
        # notifier_name should not start with a numeric char
        if isinstance(self.id, int) or self.id[0].isnumeric():
            notifier_name = ["#id" + str(self.id)]
        else:
            notifier_name = [self.id]
        notifier_name.append(self.model_args.model_name_or_path)
        if self.name:
            notifier_name.append(self.name)
        for notifier in self._notifiers.values():
            if not notifier.title:
                notifier.title = ", ".join(notifier_name)

        if has_restarts:
            for notifier in self._notifiers.values():
                notifier.notify_start(
                    message=f"Training setup ({len(restarts)} runs):", **self._train_run_args.to_sanitized_dict()
                )

        for i, seed in enumerate(restarts, start=first_run_index):
            # Set seed
            seed = set_seed(seed)
            all_results["seeds"].append(seed)

            # Set up model
            is_full_finetuning = not self.model_args.train_adapter and self.model_args.train_adapter_fusion is None
            self.model_instance = create_model(
                self.model_args,
                self.dataset_manager,
                use_classic_model_class=is_full_finetuning,
            )

            # Configure and run training
            if self._train_run_args:
                train_run_args = self._prepare_run_args(self._train_run_args, restart=i if has_restarts else None)
                trainer = Trainer(
                    self.model_instance,
                    train_run_args,
                    self.dataset_manager,
                    do_save_full_model=is_full_finetuning,
                    do_save_adapters=self.model_args.train_adapter,
                    do_save_adapter_fusion=self.model_args.train_adapter_fusion is not None,
                )
                if not has_restarts:
                    for notifier in self._notifiers.values():
                        notifier.notify_start(
                            message="Training setup:", seed=seed, **train_run_args.to_sanitized_dict()
                        )
                try:
                    step, epoch, loss, best_score, best_model_dir = trainer.train(
                        self.model_args.model_name_or_path
                        if os.path.isdir(self.model_args.model_name_or_path)
                        else None
                    )
                    # only save the final model if we don't use early stopping
                    if train_run_args.patience <= 0:
                        trainer.save_model()
                except Exception as ex:
                    for notifier in self._notifiers.values():
                        notifier.notify_error(f"{ex.__class__.__name__}: {ex}")
                    raise ex
                # if no evaluation is done, we're at the end here
                if not self._do_eval:
                    for notifier in self._notifiers.values():
                        notifier.notify_end(
                            message="Training results:",
                            step=step,
                            training_epochs=epoch,
                            loss=loss,
                            best_score=best_score,
                        )
                # otherwise, reload the best model for evaluation
                elif best_model_dir:
                    logger.info("Reloading best model for evaluation.")
                    if self.model_args.train_adapter:
                        for adapter_name in self.model_instance.config.adapters.adapters:
                            path = os.path.join(best_model_dir, adapter_name)
                            self.model_instance.load_adapter(path)
                    if self.model_args.train_adapter_fusion is not None:
                        path = os.path.join(best_model_dir, self.model_args.train_adapter_fusion)
                        # HACK: adapter-transformers refuses to overwrite existing adapter_fusion config
                        del self.model_instance.config.adapter_fusion
                        self.model_instance.load_adapter_fusion(path)
                        # HACK: also reload the prediction head
                        head_path = os.path.join(best_model_dir, self.dataset_manager.name)
                        self.model_instance.load_head(head_path)
                    if is_full_finetuning:
                        self.model_instance = self.model_instance.from_pretrained(best_model_dir)
                        register_heads(self.model_instance)
                        self.model_instance.active_head = self.dataset_manager.name
            else:
                epoch = None

            # Configure and run eval
            if self._do_eval:
                eval_run_args = self._prepare_run_args(
                    self._eval_run_args or self._train_run_args, restart=i if has_restarts else None
                )
                trainer = Trainer(
                    self.model_instance,
                    eval_run_args,
                    self.dataset_manager,
                    do_save_full_model=is_full_finetuning,
                    do_save_adapters=self.model_args.train_adapter,
                    do_save_adapter_fusion=self.model_args.train_adapter_fusion is not None,
                )
                try:
                    eval_dataset = self.dataset_manager.dataset[self._eval_split] if self._eval_split else None
                    results = trainer.evaluate(eval_dataset=eval_dataset, log=False)
                except Exception as ex:
                    for notifier in self._notifiers.values():
                        notifier.notify_error(f"{ex.__class__.__name__}: {ex}")
                    raise ex
                if epoch:
                    results["training_epochs"] = epoch
                if self._eval_split:
                    results["eval_split"] = self._eval_split
                output_eval_file = os.path.join(eval_run_args.output_dir, "eval_results.txt")
                with open(output_eval_file, "w") as f:
                    logger.info("***** Eval results {} (restart {}) *****".format(self.name, i))
                    for key, value in results.items():
                        logger.info("  %s = %s", key, value)
                        f.write("%s = %s\n" % (key, value))
                        # also append to aggregated results
                        all_results[key].append(value)
                if not has_restarts:
                    for notifier in self._notifiers.values():
                        notifier.notify_end(message="Evaluation results:", **results)
            # Delete model
            del self.model_instance

        # post-process aggregated results
        if has_restarts and self._do_eval:
            stats = {"seeds": all_results["seeds"]}
            for key, values in all_results.items():
                if isinstance(values[0], float):
                    stats[f"{key}_avg"] = np.mean(values)
                    stats[f"{key}_std"] = np.std(values)

            output_dir = self._eval_run_args.output_dir if self._eval_run_args else self._train_run_args.output_dir
            with open(os.path.join(output_dir, f"aggregated_results_{self.id}.json"), "w") as f:
                json.dump(stats, f)
            for notifier in self._notifiers.values():
                notifier.notify_end(message=f"Aggregated results ({len(restarts)} runs):", **stats)

        return dict(all_results)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Simple tool to setup Transformers training runs.")
    parser.add_argument("config", type=str, help="Path to the json file containing the full training setup.")
    parser.add_argument("--id", type=int, default=0, help="ID of this run.")
    parser.add_argument(
        "--preprocess_only", action="store_true", default=False, help="Only run dataset preprocessing."
    )
    restarts_group = parser.add_mutually_exclusive_group()
    restarts_group.add_argument("--seeds", type=lambda s: [int(item) for item in s.split(",")])
    restarts_group.add_argument("--restarts", type=int, default=None)
    # add arguments from dataclasses
    for dtype in (DatasetArguments, ModelArguments, RunArguments):
        for field in dataclasses.fields(dtype):
            field_name = f"--{field.name}"
            kwargs = field.metadata.copy()
            kwargs["type"] = field.type
            kwargs["default"] = None
            parser.add_argument(field_name, **kwargs)

    args = vars(parser.parse_args())

    # Load and run
    setup = Setup(id=args.pop("id"))
    setup.load_from_file(args.pop("config"), overrides=args)
    if args.pop("preprocess_only"):
        setup._setup_tokenizer()
        setup.dataset_manager.load_and_preprocess(CacheMode.USE_DATASET_NEW_FEATURES)
    else:
        setup.run(restarts=args.pop("seeds") or args.pop("restarts"))


if __name__ == "__main__":
    main()
