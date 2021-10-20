import logging

import numpy as np
from transformers import PreTrainedTokenizerBase

from ..arguments import DatasetArguments
from .dataset_manager import ColumnConfig, DatasetManagerBase
from .sampler import StratifiedRandomSampler


logger = logging.getLogger(__name__)


class MultipleChoiceDatasetManager(DatasetManagerBase):
    """
    Base dataset manager for multiple-choice tasks.
    """

    def __init__(self, args: DatasetArguments, tokenizer: PreTrainedTokenizerBase = None):
        super().__init__(args, tokenizer, load_metric=False)

    def train_sampler(self):
        if self.args.train_subset_size <= 0 or self.args.train_subset_size > len(self.train_split):
            return super().train_sampler()
        else:
            return StratifiedRandomSampler(
                self.train_split[self.column_config.label],
                self.args.train_subset_size,
            )

    def _custom_filter(self, example):
        if self.column_config.label:
            label = example[self.column_config.label]
            if isinstance(label, str):
                label = label.strip(" \n")
            valid_label = label in self.choice_label_map
        else:
            valid_label = True
        return valid_label and all([example[col] is not None for col in self.column_config.inputs])

    def _build_input_choice(self, question, ending):
        if "_" in question:
            return question.replace("_", ending)
        else:
            return question + " " + ending

    def _build_inputs(self, example):
        context, question, endings = example
        a_s = [context for _ in range(len(self.choice_label_map))]
        b_s = [self._build_input_choice(question, endings[i]) for i in range(len(self.choice_label_map))]
        return a_s, b_s

    def encode_batch(self, examples):
        input_ids = []
        token_type_ids = []
        attention_mask = []
        for example in zip(*[examples[c] for c in self.column_config.inputs]):
            a_s, b_s = self._build_inputs(example)
            encoded = self.tokenizer(
                a_s,
                b_s,
                max_length=self.args.max_seq_length,
                truncation=self._truncation,
                padding=self._padding,
                return_overflowing_tokens=True,
            )
            # if "overflowing_tokens" in encoded and len(encoded["overflowing_tokens"][0]) > 0:
            #     logger.info("Cropping {0} tokens of input.".format(len(encoded["overflowing_tokens"][0])))
            input_ids.append(encoded["input_ids"])
            if "token_type_ids" in encoded:
                token_type_ids.append(encoded["token_type_ids"])
            if "attention_mask" in encoded:
                attention_mask.append(encoded["attention_mask"])
        labels = []
        for label in examples[self.column_config.label]:
            if isinstance(label, str):
                label = label.strip(" \n")
            labels.append(self.choice_label_map.get(label, None))
        encoded = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if len(token_type_ids) > 0:
            encoded["token_type_ids"] = token_type_ids
        return encoded

    def compute_metrics(self, predictions, references):
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = np.argmax(predictions, axis=1)
        return {"accuracy": (predictions == references).mean()}

    def get_prediction_head_config(self):
        return {
            "head_type": "multiple_choice",
            "num_choices": len(self.choice_label_map),
            "layers": 2,
            "activation_function": "tanh",
        }


class COPAManager(MultipleChoiceDatasetManager):
    _COPA_DICT = {
        "cause": "What was the cause of this?",
        "effect": "What happened as a result?",
    }

    def __init__(self, args: DatasetArguments, tokenizer: PreTrainedTokenizerBase = None):
        super().__init__(args, tokenizer)
        self.column_config = ColumnConfig(["premise", "alternative1", "alternative2", "relation"], "label")
        self.choice_label_map = {v: k for (k, v) in enumerate(["1", "2"])}

    def _build_inputs(self, example):
        premise, alternative1, alternative2, relation = example
        a_s = [premise for _ in range(len(self.choice_label_map))]
        b_s = [self._COPA_DICT[relation] + " " + alt for alt in [alternative1, alternative2]]
        return a_s, b_s


class SWAGManager(MultipleChoiceDatasetManager):
    def __init__(self, args: DatasetArguments, tokenizer: PreTrainedTokenizerBase = None):
        super().__init__(args, tokenizer)
        self._use_task_name_for_loading = False
        self._default_subset_name = "regular"
        self.column_config = ColumnConfig(["sent1", "sent2", "ending0", "ending1", "ending2", "ending3"], "label")
        self.choice_label_map = {v: k for (k, v) in enumerate(range(4))}

    def _build_inputs(self, example):
        a_s = [example[0] for _ in range(4)]
        b_s = [
            example[1] + " " + example[2],
            example[1] + " " + example[3],
            example[1] + " " + example[4],
            example[1] + " " + example[5],
        ]
        return a_s, b_s


class CosmosQAManager(MultipleChoiceDatasetManager):
    def __init__(self, args: DatasetArguments, tokenizer: PreTrainedTokenizerBase = None):
        super().__init__(args, tokenizer)
        self.column_config = ColumnConfig(["context", "question", "answer0", "answer1", "answer2", "answer3"], "label")
        self.choice_label_map = {v: k for (k, v) in enumerate(range(4))}

    def _build_inputs(self, example):
        a_s = [example[0] for _ in range(4)]
        b_s = [
            example[1] + " " + example[2],
            example[1] + " " + example[3],
            example[1] + " " + example[4],
            example[1] + " " + example[5],
        ]
        return a_s, b_s


class HellaswagManager(MultipleChoiceDatasetManager):
    def __init__(self, args: DatasetArguments, tokenizer: PreTrainedTokenizerBase = None):
        super().__init__(args, tokenizer)
        self.column_config = ColumnConfig(["ctx_a", "ctx_b", "endings"], "label")
        self.choice_label_map = {v: k for (k, v) in enumerate(["0", "1", "2", "3"])}


class RaceManager(MultipleChoiceDatasetManager):
    def __init__(self, args: DatasetArguments, tokenizer: PreTrainedTokenizerBase = None):
        super().__init__(args, tokenizer)
        self.column_config = ColumnConfig(["article", "question", "options"], "answer")
        self.choice_label_map = {v: k for (k, v) in enumerate(["A", "B", "C", "D"])}


class QuailManager(MultipleChoiceDatasetManager):
    def __init__(self, args: DatasetArguments, tokenizer: PreTrainedTokenizerBase = None):
        super().__init__(args, tokenizer)
        self.column_config = ColumnConfig(["context", "question", "answers"], "correct_answer_id")
        self.choice_label_map = {v: k for (k, v) in enumerate(range(4))}


class ARTManager(MultipleChoiceDatasetManager):
    def __init__(self, args: DatasetArguments, tokenizer: PreTrainedTokenizerBase = None):
        super().__init__(args, tokenizer)
        self.column_config = ColumnConfig(["observation_1", "observation_2", "hypothesis_1", "hypothesis_2"], "label")
        self.choice_label_map = {v: k for (k, v) in enumerate([1, 2])}

    def _build_inputs(self, example):
        a_s = [example[0] + " " + example[2], example[0] + " " + example[3]]
        b_s = [example[1] for _ in range(2)]
        return a_s, b_s


class SocialIQAManager(MultipleChoiceDatasetManager):
    def __init__(self, args: DatasetArguments, tokenizer: PreTrainedTokenizerBase = None):
        super().__init__(args, tokenizer)
        self.column_config = ColumnConfig(["context", "question", "answerA", "answerB", "answerC"], "label")
        self.choice_label_map = {v: k for (k, v) in enumerate(["1", "2", "3"])}

    def _build_inputs(self, example):
        a_s = [example[0] for _ in range(3)]
        b_s = [
            example[1] + " " + example[2],
            example[1] + " " + example[3],
            example[1] + " " + example[4],
        ]
        return a_s, b_s


class CommonsenseQAManager(MultipleChoiceDatasetManager):
    def __init__(self, args: DatasetArguments, tokenizer: PreTrainedTokenizerBase = None):
        super().__init__(args, tokenizer=tokenizer)
        self.column_config = ColumnConfig(["question", "choices"], "answerKey")
        self.choice_label_map = {v: k for (k, v) in enumerate(["A", "B", "C", "D", "E"])}

    def _build_inputs(self, example):
        question, choices = example
        a_s = [question for _ in range(len(self.choice_label_map))]
        b_s = [choices["text"][i] for i in range(len(self.choice_label_map))]
        return a_s, b_s


class QuartzManager(MultipleChoiceDatasetManager):
    def __init__(self, args: DatasetArguments, tokenizer: PreTrainedTokenizerBase = None):
        super().__init__(args, tokenizer=tokenizer)
        self.column_config = ColumnConfig(["para", "question", "choices"], "answerKey")
        self.choice_label_map = {v: k for (k, v) in enumerate(["A", "B"])}

    def _build_input_choice(self, question, ending):
        if "_____" in question:
            return question.replace("_____", ending)
        else:
            return question + " " + ending

    def _build_inputs(self, example):
        para, question, choices = example
        a_s = [para for _ in range(len(self.choice_label_map))]
        b_s = [self._build_input_choice(question, choices["text"][i]) for i in range(len(self.choice_label_map))]
        return a_s, b_s


class WinograndeManager(MultipleChoiceDatasetManager):
    def __init__(self, args: DatasetArguments, tokenizer: PreTrainedTokenizerBase = None):
        super().__init__(args, tokenizer)
        self.column_config = ColumnConfig(["sentence", "option1", "option2"], "answer")
        self.choice_label_map = {v: k for (k, v) in enumerate(["1", "2"])}

    def _build_inputs(self, example):
        sentence, opt1, opt2 = example
        a_s = [self._build_input_choice(sentence, opt) for opt in [opt1, opt2]]
        return a_s, None
