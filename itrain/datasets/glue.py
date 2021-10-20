import numpy as np
from transformers import PreTrainedTokenizerBase

from ..arguments import DatasetArguments
from .classification import ClassificationDatasetManager
from .dataset_manager import ColumnConfig


class GlueManager(ClassificationDatasetManager):
    """
    Dataset manager for GLUE benchmark.
    """

    tasks_num_labels = {
        "cola": 2,
        "mnli": 3,
        "mrpc": 2,
        "sst2": 2,
        "stsb": 1,
        "qqp": 2,
        "qnli": 2,
        "rte": 2,
        "wnli": 2,
    }

    def __init__(self, args: DatasetArguments, tokenizer: PreTrainedTokenizerBase = None):
        super().__init__(args, tokenizer, load_metric=True)

    def _configure(self):
        if (
            self.args.task_name == "mrpc"
            or self.args.task_name == "rte"
            or self.args.task_name == "wnli"
            or self.args.task_name == "stsb"
        ):
            self.column_config = ColumnConfig(["sentence1", "sentence2"], "label")
        elif self.args.task_name == "sst2":
            self.column_config = ColumnConfig(["sentence"], "label")
        elif self.args.task_name == "cola":
            self.column_config = ColumnConfig(["sentence"], "label")
        elif self.args.task_name == "qqp":
            self.column_config = ColumnConfig(["question1", "question2"], "label")
        elif self.args.task_name == "mnli":
            self.column_config = ColumnConfig(["premise", "hypothesis"], "label")
            self.dev_split_name = "validation_matched"
            self.test_split_name = "test_matched"
        elif self.args.task_name == "qnli":
            self.column_config = ColumnConfig(["question", "sentence"], "label")
        else:
            raise ValueError()

    def compute_metrics(self, predictions, references):
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        if self.args.task_name == "stsb":
            predictions = np.squeeze(predictions)
        else:
            predictions = np.argmax(predictions, axis=1)
        return self.metric.compute(predictions=predictions, references=references)
