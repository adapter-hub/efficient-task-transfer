from collections import defaultdict

import numpy as np
from transformers import PreTrainedTokenizerBase

from ..arguments import DatasetArguments
from ..ext.super_glue import SuperGlue
from .classification import ClassificationDatasetManager
from .dataset_manager import ColumnConfig


class SuperGlueManager(ClassificationDatasetManager):
    """
    Dataset manager for SuperGLUE benchmark.
    """

    tasks_num_labels = {
        "boolq": 2,
        "cb": 3,
        "copa": 2,
        "multirc": 2,
        "record": 2,
        "rte": 2,
        "wic": 2,
        "wsc": 2,
        "wsc.fixed": 2,
        "axb": 2,
        "axg": 2,
    }
    _COPA_DICT = {
        "cause": "What was the cause of this?",
        "effect": "What happened as a result?",
    }

    def __init__(self, args: DatasetArguments, tokenizer: PreTrainedTokenizerBase = None):
        super().__init__(args, tokenizer, load_metric=SuperGlue)

    def _configure(self):
        if self.args.task_name == "boolq":
            self.column_config = ColumnConfig(["passage", "question"], "label")
            self._encode_batch = self._encode_batch_classification
        elif self.args.task_name == "cb" or self.args.task_name == "rte" or self.args.task_name == "axg":
            self.column_config = ColumnConfig(["premise", "hypothesis"], "label")
            self._encode_batch = self._encode_batch_classification
        elif self.args.task_name == "copa":
            self.column_config = ColumnConfig(["premise", "question", "choice1", "choice2"], "label")
            self._encode_batch = self._encode_batch_copa
        elif self.args.task_name == "multirc":
            self.column_config = ColumnConfig(["paragraph", "question", "answer"], "label")
            self._encode_batch = self._encode_batch_multirc
        elif self.args.task_name == "axb":
            self.column_config = ColumnConfig(["sentence1", "sentence2"], "label")
            self._encode_batch = self._encode_batch_classification
        elif self.args.task_name == "record":
            self.column_config = ColumnConfig(["passage", "query", "entities"], "answers")
            self._encode_batch = self._encode_batch_record
            self._no_stratification = True
        elif self.args.task_name == "wic":
            self.column_config = ColumnConfig(["sentence1", "sentence2", "word"], "label")
            self._encode_batch = self._encode_batch_wic
        # TODO wsc ?
        else:
            raise ValueError()

    def _custom_filter(self, example):
        return True  # Override for custom filtering

    def encode_batch(self, examples):
        return self._encode_batch(examples)

    def _encode_batch_copa(self, examples):
        contexts = [p + " " + self._COPA_DICT[q] for p, q in zip(examples["premise"], examples["question"])]
        sentences_a = [ctx + " " + choice for ctx, choice in zip(contexts, examples["choice1"])]
        sentences_b = [ctx + " " + choice for ctx, choice in zip(contexts, examples["choice2"])]
        encoded = self.tokenizer(
            sentences_a,
            sentences_b,
            max_length=self.args.max_seq_length,
            truncation=self._truncation,
            padding=self._padding,
        )
        encoded.update({"labels": examples[self.column_config.label]})
        return encoded

    def _encode_batch_multirc(self, examples):
        contexts = [
            paragraph + " " + question for paragraph, question in zip(examples["paragraph"], examples["question"])
        ]
        encoded = self.tokenizer(
            contexts,
            examples["answer"],
            max_length=self.args.max_seq_length,
            truncation=self._truncation,
            padding=self._padding,
        )
        encoded.update({"labels": examples[self.column_config.label]})
        return encoded

    def _encode_batch_record(self, examples):
        encoded = defaultdict(list)
        for idx, passage, query, entities, answers in zip(
            examples["idx"], examples["passage"], examples["query"], examples["entities"], examples["answers"]
        ):
            for entity in entities:
                label = 1 if entity in answers else 0
                query_filled = query.replace("@placeholder", entity)
                example_encoded = self.tokenizer(
                    passage,
                    query_filled,
                    max_length=self.args.max_seq_length,
                    truncation=self._truncation,
                    padding=self._padding,
                    return_overflowing_tokens=True,
                )
                # if "overflowing_tokens" in example_encoded and len(example_encoded["overflowing_tokens"]) > 0:
                #     logger.info("Cropping {0} tokens of input.".format(len(example_encoded["overflowing_tokens"])))
                encoded["idx"].append(idx)
                encoded["passage"].append(passage)
                encoded["query"].append(query_filled)
                encoded["entities"].append(entity)
                encoded["answers"].append(answers)
                encoded["input_ids"].append(example_encoded["input_ids"])
                encoded["labels"].append(label)
                if "token_type_ids" in example_encoded:
                    encoded["token_type_ids"].append(example_encoded["token_type_ids"])
                if "attention_mask" in example_encoded:
                    encoded["attention_mask"].append(example_encoded["attention_mask"])
        return encoded

    def _encode_batch_wic(self, examples):
        sentences = []
        # concat all three input parts with [SEP] token
        for parts in zip(*[examples[c] for c in self.column_config.inputs]):
            sentences.append(self.tokenizer.sep_token.join(parts))
        encoded = self.tokenizer(
            sentences,
            max_length=self.args.max_seq_length,
            truncation=self._truncation,
            padding=self._padding,
        )
        encoded.update({"labels": examples[self.column_config.label]})
        return encoded

    def _encode_batch_wsc(self, examples):
        raise NotImplementedError()  # TODO

    def _encode_batch_classification(self, examples):
        encoded = self.tokenizer(
            examples[self.column_config.inputs[0]],
            examples[self.column_config.inputs[1]] if len(self.column_config.inputs) > 1 else None,
            max_length=self.args.max_seq_length,
            truncation=self._truncation,
            padding=self._padding,
        )
        encoded.update({"labels": examples[self.column_config.label]})
        return encoded

    def compute_metrics(self, predictions, references):
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        if self.args.task_name == "multirc":
            predictions = np.argmax(predictions, axis=1)
            predictions = [{"idx": idx, "prediction": pred} for idx, pred in zip(self.dev_split["idx"], predictions)]
        elif self.args.task_name == "record":
            max_preds = {}  # group predictions by question id
            for idx, entity, pred, answers in zip(
                self.dev_split["idx"], self.dev_split["entities"], predictions, self.dev_split["answers"]
            ):
                idx_string = f"{idx['passage']}-{idx['query']}"
                if idx_string not in max_preds or pred[1] > max_preds[idx_string]["logit"]:
                    max_preds[idx_string] = {"idx": idx, "logit": pred[1], "entity": entity, "answers": answers}
            predictions = [{"idx": val["idx"], "prediction_text": val["entity"]} for _, val in max_preds.items()]
            references = [{"idx": val["idx"], "answers": val["answers"]} for _, val in max_preds.items()]
        else:
            predictions = np.argmax(predictions, axis=1)
        return self.metric.compute(predictions=predictions, references=references)
