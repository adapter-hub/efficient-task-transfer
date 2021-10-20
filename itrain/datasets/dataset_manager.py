import inspect
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Union

from datasets import GenerateMode, Metric, Split, load_dataset, load_metric
from transformers import PreTrainedTokenizerBase, default_data_collator
from transformers.file_utils import torch_cache_home

from ..arguments import DatasetArguments
from .sampler import RandomSampler


logger = logging.getLogger(__name__)

DATASET_FEATURES_CACHE = os.path.join(torch_cache_home, "itrain")
if not os.path.exists(DATASET_FEATURES_CACHE):
    os.makedirs(DATASET_FEATURES_CACHE)


class CacheMode(IntEnum):
    NEW_DATASET_NEW_FEATURES = 0
    USE_DATASET_NEW_FEATURES = 1
    USE_DATASET_USE_FEATURES = 2


@dataclass
class ColumnConfig:
    inputs: List[str]
    label: str


class DatasetManager(ABC):
    """
    Abstract base class of all dataset manager implementations.
    All dataset implementations should at least inherit from this class.
    """

    label_column_names = ["labels"]

    def __init__(
        self,
        args: DatasetArguments,
        tokenizer: PreTrainedTokenizerBase = None,
    ):
        self.args = args
        if self.args.dataset_dir:
            self._dataset_loc = os.path.expanduser(self.args.dataset_dir)
        else:
            self._dataset_loc = self.args.dataset_name
        self.tokenizer = tokenizer
        self.dataset = {}
        self.metric = None
        self.train_split_name = Split.TRAIN
        self.dev_split_name = Split.VALIDATION
        self.test_split_name = Split.TEST
        self.always_call_metrics = False  # a bit hacky but ensures metrics are computed for qa

    @property
    def name(self):
        """The full name of the dataset."""
        if self.args.task_name:
            name = f"{self.args.dataset_name}_{self.args.task_name}"
        else:
            name = self.args.dataset_name
        if self.args.train_subset_size > 0:
            name = f"{name}_n{self.args.train_subset_size}"
        return name

    def _get_features_cache_file(self, split_name):
        return os.path.join(
            DATASET_FEATURES_CACHE,
            "_".join([self.tokenizer.__class__.__name__, self.args.identifier, split_name]) + ".cached",
        )

    @abstractmethod
    def load(self, cache_mode: CacheMode = CacheMode.USE_DATASET_USE_FEATURES):
        """
        Loads all dataset data into the manager instance. Optionally downloads the dataset and caches it.
        preprocess() should be called after this method to convert the loaded data into the desired format for training.
        """
        pass

    @abstractmethod
    def preprocess(self, cache_mode: CacheMode = CacheMode.USE_DATASET_USE_FEATURES):
        """
        Preprocesses the loaded dataset into the desired format for training.
        The dataset has to be loaded using load() before this method is called.
        """
        pass

    def load_and_preprocess(self, cache_mode: CacheMode = CacheMode.USE_DATASET_USE_FEATURES):
        """
        Loads and preprocesses the dataset. Equivalent to calling load() and then preprocess().
        """
        self.load(cache_mode)
        self.preprocess(cache_mode)

    def train_sampler(self):
        return RandomSampler(self.train_split, num_samples=self.args.train_subset_size)

    def collate_fn(self, features):
        return default_data_collator(features)

    @abstractmethod
    def compute_metrics(self, predictions, references):
        pass

    @property
    def train_split(self):
        return self.dataset[self.train_split_name] if self.train_split_name in self.dataset else None

    @property
    def dev_split(self):
        return self.dataset[self.dev_split_name] if self.dev_split_name in self.dataset else None

    @property
    def test_split(self):
        return self.dataset[self.test_split_name] if self.test_split_name in self.dataset else None

    def get_tokenizer_config_kwargs(self):
        return {}

    def get_model_config_kwargs(self):
        return {}

    @abstractmethod
    def get_prediction_head_config(self):
        pass


class DatasetManagerBase(DatasetManager):
    def __init__(
        self,
        args: DatasetArguments,
        tokenizer: PreTrainedTokenizerBase = None,
        load_metric: Union[bool, Metric] = True,
    ):
        super().__init__(args, tokenizer)
        self._use_task_name_for_loading = True
        self._default_subset_name = None
        self._padding = "max_length"
        self._truncation = True
        self._encode_remove_columns = False
        self.load_metric = load_metric

    def load(self, cache_mode: CacheMode = CacheMode.USE_DATASET_USE_FEATURES):
        # load dataset
        download_mode = (
            GenerateMode.FORCE_REDOWNLOAD
            if cache_mode == CacheMode.NEW_DATASET_NEW_FEATURES
            else GenerateMode.REUSE_DATASET_IF_EXISTS
        )
        self.dataset = load_dataset(
            self._dataset_loc,
            self.args.task_name if self._use_task_name_for_loading else self._default_subset_name,
            download_mode=download_mode,
        )
        # split a validation set from the training set if not present
        if self.dev_split_name not in self.dataset:
            logger.warning("Dataset does not have a pre-defined validation split. Creating one from the training set.")
            train_dev_split = self.dataset[self.train_split_name].train_test_split(test_size=0.2)
            self.dataset[str(self.train_split_name)] = train_dev_split["train"]
            self.dataset[str(self.dev_split_name)] = train_dev_split["test"]

        # load metric
        if self.load_metric:
            if inspect.isclass(self.load_metric) and issubclass(self.load_metric, Metric):
                self.metric = self.load_metric(self.args.task_name)
            else:
                self.metric = load_metric(self.args.dataset_name, self.args.task_name)

    def preprocess(self, cache_mode: CacheMode = CacheMode.USE_DATASET_USE_FEATURES):
        # filter
        self.dataset[self.train_split_name] = self.dataset[self.train_split_name].filter(self._custom_filter)
        if self.dev_split_name in self.dataset:
            self.dataset[self.dev_split_name] = self.dataset[self.dev_split_name].filter(self._custom_filter)
        if (
            self.dataset.num_rows[self.train_split_name] == 0
            or self.dev_split_name in self.dataset
            and self.dataset.num_rows[self.dev_split_name] == 0
        ):
            raise ValueError("No examples left in at least one split of the dataset after filtering.")
        # convert examples to transformers features
        self._encode(load_from_cache=cache_mode >= CacheMode.USE_DATASET_USE_FEATURES)

    def _custom_filter(self, example):
        return True  # Override for custom filtering

    @abstractmethod
    def encode_batch(self, examples):
        pass

    def _encode(self, load_from_cache=True):
        cache_files = {}
        for split_name in self.dataset.keys():
            cache_files[split_name] = self._get_features_cache_file(split_name)
        # HACK: currently expects columns in all splits to be identical
        remove_columns = self.dataset.column_names[self.train_split_name] if self._encode_remove_columns else None
        self.dataset = self.dataset.map(
            self.encode_batch,
            batched=True,
            cache_file_names=cache_files,
            load_from_cache_file=load_from_cache,
            remove_columns=remove_columns,
        )
        self.dataset.set_format(columns=["input_ids"] + self.label_column_names + self.tokenizer.model_input_names)

    def collate_fn(self, features):
        return default_data_collator(features)

    def compute_metrics(self, predictions, references):
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        return self.metric.compute(predictions=predictions, references=references)
