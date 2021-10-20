"""
Code taken and modified from: https://github.com/Adapter-Hub/hgiyt.
Credits: P. Rust et al.
"""
from collections import defaultdict

import numpy as np
from transformers import PreTrainedTokenizerBase

from ..arguments import DatasetArguments
from .dataset_manager import DatasetManagerBase


UD_HEAD_LABELS = [
    "_",
    "acl",
    "advcl",
    "advmod",
    "amod",
    "appos",
    "aux",
    "case",
    "cc",
    "ccomp",
    "clf",
    "compound",
    "conj",
    "cop",
    "csubj",
    "dep",
    "det",
    "discourse",
    "dislocated",
    "expl",
    "fixed",
    "flat",
    "goeswith",
    "iobj",
    "list",
    "mark",
    "nmod",
    "nsubj",
    "nummod",
    "obj",
    "obl",
    "orphan",
    "parataxis",
    "punct",
    "reparandum",
    "root",
    "vocative",
    "xcomp",
]


class ParsingMetric:
    """
    Based on allennlp.training.metrics.AttachmentScores
    Computes labeled and unlabeled attachment scores for a dependency parse. Note that the input
    to this metric is the sampled predictions, not the distribution itself.
    """

    def __init__(self):
        self._labeled_correct = 0.0
        self._unlabeled_correct = 0.0
        self._total_words = 0.0

    def add(
        self,
        gold_indices: np.ndarray,
        gold_labels: np.ndarray,
        predicted_indices: np.ndarray,
        predicted_labels: np.ndarray,
    ):
        predicted_indices = predicted_indices.astype(np.long)
        predicted_labels = predicted_labels.astype(np.long)
        gold_indices = gold_indices.astype(np.long)
        gold_labels = gold_labels.astype(np.long)

        correct_indices = (predicted_indices == gold_indices).astype(np.long)
        correct_labels = (predicted_labels == gold_labels).astype(np.long)
        correct_labels_and_indices = correct_indices * correct_labels

        self._unlabeled_correct += correct_indices.sum()
        self._labeled_correct += correct_labels_and_indices.sum()
        self._total_words += correct_indices.size

    def get_metric(self):
        unlabeled_attachment_score = 0.0
        labeled_attachment_score = 0.0
        if self._total_words > 0.0:
            unlabeled_attachment_score = self._unlabeled_correct / self._total_words
            labeled_attachment_score = self._labeled_correct / self._total_words
        return {
            "uas": unlabeled_attachment_score * 100,
            "las": labeled_attachment_score * 100,
        }

    def reset(self):
        self._labeled_correct = 0.0
        self._unlabeled_correct = 0.0
        self._total_words = 0.0


class UniversalDependenciesManager(DatasetManagerBase):
    label_column_names = ["labels_rels", "labels_arcs", "word_starts"]

    def __init__(self, args: DatasetArguments, tokenizer: PreTrainedTokenizerBase = None):
        super().__init__(args, tokenizer, load_metric=False)
        self.label_list = UD_HEAD_LABELS
        self.label_map = {label: i for i, label in enumerate(self.label_list)}
        self._pad_token = -1
        self._encode_remove_columns = True

    def encode_batch(self, examples):
        features = defaultdict(list)
        for words, heads, deprels in zip(examples["tokens"], examples["head"], examples["deprel"]):
            # clean up
            i = 0
            while i < len(heads):
                if heads[i] == "None":
                    del words[i]
                    del heads[i]
                    del deprels[i]
                i += 1
            tokens = [self.tokenizer.tokenize(w) for w in words]
            word_lengths = [len(w) for w in tokens]
            tokens_merged = []
            list(map(tokens_merged.extend, tokens))

            if 0 in word_lengths:
                continue
            # Filter out sequences that are too long
            if len(tokens_merged) >= (self.args.max_seq_length - 2):
                continue

            encoding = self.tokenizer(
                words,
                add_special_tokens=True,
                padding=self._padding,
                truncation=self._truncation,
                max_length=self.args.max_seq_length,
                is_split_into_words=True,
                return_token_type_ids=True,
                return_attention_mask=True,
            )

            input_ids = encoding["input_ids"]
            token_type_ids = encoding["token_type_ids"]
            attention_mask = encoding["attention_mask"]

            pad_item = [self._pad_token]

            # pad or truncate arc labels
            labels_arcs = [int(h) for h in heads]
            labels_arcs = labels_arcs + (self.args.max_seq_length - len(labels_arcs)) * pad_item

            # convert rel labels from map, pad or truncate if necessary
            labels_rels = [self.label_map[i.split(":")[0]] for i in deprels]
            labels_rels = labels_rels + (self.args.max_seq_length - len(labels_rels)) * pad_item

            # determine start indices of words, pad or truncate if necessary
            word_starts = np.cumsum([1] + word_lengths).tolist()
            word_starts = word_starts + (self.args.max_seq_length + 1 - len(word_starts)) * pad_item

            # sanity check lengths
            assert len(input_ids) == self.args.max_seq_length
            assert len(attention_mask) == self.args.max_seq_length
            assert len(token_type_ids) == self.args.max_seq_length
            assert len(labels_arcs) == self.args.max_seq_length
            assert len(labels_rels) == self.args.max_seq_length
            assert len(word_starts) == self.args.max_seq_length + 1

            features["input_ids"].append(input_ids)
            features["attention_mask"].append(attention_mask)
            features["token_type_ids"].append(token_type_ids)
            features["word_starts"].append(word_starts)
            features["labels_arcs"].append(labels_arcs)
            features["labels_rels"].append(labels_rels)

        return dict(features)

    def compute_metrics(self, predictions, references):
        metric = ParsingMetric()

        for rel_preds, arc_preds, labels_rels, labels_arcs, _ in zip(*predictions, *references):

            mask = np.not_equal(labels_arcs, self._pad_token)
            predictions_arcs = np.argmax(arc_preds, axis=-1)[mask]

            labels_arcs = labels_arcs[mask]

            predictions_rels, labels_rels = rel_preds[mask], labels_rels[mask]
            predictions_rels = predictions_rels[np.arange(len(labels_arcs)), labels_arcs]
            predictions_rels = np.argmax(predictions_rels, axis=-1)

            metric.add(labels_arcs, labels_rels, predictions_arcs, predictions_rels)

        results = metric.get_metric()
        return results

    def get_tokenizer_config_kwargs(self):
        return {"add_prefix_space": True}

    def get_model_config_kwargs(self):
        return {"pad_token_id": self._pad_token}

    def get_prediction_head_config(self):
        return {
            "head_type": "dependency_parsing",
            "num_labels": len(self.label_list),
            "label2id": self.label_map,
        }
