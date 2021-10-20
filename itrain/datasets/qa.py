import logging
import os
import time

import torch
from datasets import DownloadConfig, DownloadManager
from filelock import FileLock
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase, SquadDataset, SquadDataTrainingArguments
from transformers.data.datasets.squad import Split as SquadSplit
from transformers.data.metrics.squad_metrics import compute_predictions_logits, squad_evaluate
from transformers.data.processors.squad import (
    SquadExample,
    SquadResult,
    SquadV1Processor,
    squad_convert_example_to_features,
    squad_convert_example_to_features_init,
    whitespace_tokenize,
)

from ..arguments import DatasetArguments
from .dataset_manager import DATASET_FEATURES_CACHE, CacheMode, DatasetManager
from .sampler import QAPossibleSubsetRandomSampler


logger = logging.getLogger(__name__)


DOWNLOAD_URL_BASE = "https://multiqa.s3.amazonaws.com/squad2-0_format_data/"

DATASET_DOWNLOAD_CACHE = os.path.join(DATASET_FEATURES_CACHE, "download")
if not os.path.exists(DATASET_DOWNLOAD_CACHE):
    os.makedirs(DATASET_DOWNLOAD_CACHE)


class SquadLikeDataset(SquadDataset):
    def __init__(
        self,
        args: SquadDataTrainingArguments,
        tokenizer: PreTrainedTokenizerBase,
        cached_features_file: str,
        mode: str = "train",
        data_file: str = None,
        do_lower_case=True,
    ):
        self.args = args
        # which version to use here is irrelevant as we set the file names manually below
        self.processor = SquadV1Processor()
        if isinstance(mode, str):
            try:
                mode = SquadSplit[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")
        self.mode = mode

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.old_features = torch.load(cached_features_file)

                self.features = self.old_features["features"]
                self.examples = self.old_features.get("examples", None)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                if mode == SquadSplit.dev:
                    self.examples = self.processor.get_dev_examples(args.data_dir, filename=data_file)
                    if do_lower_case:
                        for example in self.examples:
                            # lower-case all examples since there's some case mismatch between context and answer
                            example.doc_tokens = [token.lower() for token in example.doc_tokens]
                            example.question_text = example.question_text.lower()
                            example.context_text = example.context_text.lower()
                else:
                    # ugly hack: always use the dev_examples method and patch answers afterwards
                    self.examples = self.processor.get_dev_examples(args.data_dir, filename=data_file)
                    removed = 0
                    unanswerable = 0
                    example: SquadExample
                    for example in self.examples:
                        # lower-case all examples since there's some case mismatch between context and answer
                        if do_lower_case:
                            example.doc_tokens = [token.lower() for token in example.doc_tokens]
                            example.question_text = example.question_text.lower()
                            example.context_text = example.context_text.lower()
                        example.is_valid = True
                        if not example.is_impossible:
                            if len(example.answers) > 0:
                                answer = example.answers[0]
                                if do_lower_case:
                                    example.answer_text = answer["text"].lower()
                                else:
                                    example.answer_text = answer["text"]
                                example.start_position_character = answer["answer_start"]
                                # set start and end position
                                example.start_position = example.char_to_word_offset[answer["answer_start"]]
                                example.end_position = example.char_to_word_offset[
                                    min(
                                        example.start_position_character + len(example.answer_text) - 1,
                                        len(example.char_to_word_offset) - 1,
                                    )
                                ]
                                if not self._fix_answer(example):
                                    removed += 1
                            else:
                                # discard question if it has no answers and is not marked unanswerable
                                removed += 1
                                example.is_valid = False
                        else:
                            unanswerable += 1
                            if not args.version_2_with_negative:
                                logger.warn(
                                    "Found impossible question but dataset should not have them."
                                    "Set version_2_with_negative to True."
                                )
                                removed += 1
                                example.is_valid = False
                    logger.info(
                        f"Found a total of {len(self.examples)} examples, including {unanswerable} unanswerable."
                    )
                    if removed > 0:
                        logger.warn(f"Removed {removed} samples from training because of missing answers.")
                        self.examples = [example for example in self.examples if example.is_valid]

                self.features = squad_convert_examples_to_features(
                    examples=self.examples,
                    tokenizer=tokenizer,
                    max_seq_length=args.max_seq_length,
                    doc_stride=args.doc_stride,
                    max_query_length=args.max_query_length,
                    is_training=mode == SquadSplit.train,
                    # threads=args.threads,
                )

                # free some space
                if mode == SquadSplit.dev:
                    pass
                    # for example in self.examples:
                    #     del example.question_text
                    #     del example.context_text
                    #     del example.title
                else:
                    del self.examples

                start = time.time()
                torch.save(
                    {"features": self.features, "examples": getattr(self, "examples", None)},
                    cached_features_file,
                )
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def _fix_answer(self, example):
        # check if the answer span is actually valid
        actual_text = " ".join(example.doc_tokens[example.start_position : (example.end_position + 1)]).lower()
        cleaned_answer_text = " ".join(whitespace_tokenize(example.answer_text)).lower()
        if actual_text.find(cleaned_answer_text) == -1:
            # try if the span is just shifted by one to the left or right
            actual_text_left = " ".join(
                example.doc_tokens[(example.start_position - 1) : example.end_position]
            ).lower()
            actual_text_right = " ".join(
                example.doc_tokens[(example.start_position + 1) : (example.end_position + 2)]
            ).lower()
            if actual_text_left.find(cleaned_answer_text) != -1:
                example.start_position = example.start_position - 1
                example.end_position = example.end_position - 1
            elif actual_text_right.find(cleaned_answer_text) != -1:
                example.start_position = example.start_position + 1
                example.end_position = example.end_position + 1
            else:
                logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
                example.is_valid = False
                return False
        return True


# TODO: this method should not be here
def squad_convert_examples_to_features(
    examples,
    tokenizer,
    max_seq_length,
    doc_stride,
    max_query_length,
    is_training,
    padding_strategy="max_length",
    tqdm_enabled=True,
):
    squad_convert_example_to_features_init(tokenizer)

    features = []
    for example in tqdm(
        examples,
        total=len(examples),
        desc="convert squad examples to features",
        disable=not tqdm_enabled,
    ):
        features.append(
            squad_convert_example_to_features(
                example,
                max_seq_length=max_seq_length,
                doc_stride=doc_stride,
                max_query_length=max_query_length,
                padding_strategy=padding_strategy,
                is_training=is_training,
            )
        )

        # free space
        del example.question_text
        del example.context_text
        del example.title

    new_features = []
    unique_id = 1000000000
    example_index = 0
    for example_features in tqdm(
        features, total=len(features), desc="add example index and unique id", disable=not tqdm_enabled
    ):
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    features = new_features
    del new_features
    return features


class QADatasetManager(DatasetManager):
    """
    Base dataset manager for SQuAD-like extractive QA tasks.
    """

    label_column_names = ["start_positions", "end_positions"]
    with_negative = False
    train_file_name = None
    dev_file_name = None

    def __init__(self, args: DatasetArguments, tokenizer: PreTrainedTokenizerBase = None):
        super().__init__(args, tokenizer=tokenizer)
        self.train_split_name = "train"
        self.dev_split_name = "dev"
        self.always_call_metrics = True

    def train_sampler(self):
        if (
            self.with_negative
            or self.args.train_subset_size < 0
            or self.args.train_subset_size >= len(self.train_split)
        ):
            return super().train_sampler()
        else:
            # only sample from the possible questions of the dataset
            return QAPossibleSubsetRandomSampler(
                self.train_split.features,
                self.args.train_subset_size,
            )

    def _create_squad_training_arguments(self, data_dir, overwrite_cache=False):
        # copy fields from DatasetArguments
        self.squad_args = SquadDataTrainingArguments()
        self.squad_args.max_seq_length = self.args.max_seq_length
        self.squad_args.doc_stride = self.args.doc_stride
        self.squad_args.data_dir = data_dir
        self.squad_args.version_2_with_negative = self.with_negative
        self.squad_args.overwrite_cache = overwrite_cache
        # self.squad_args.threads = 1

    def load(self, cache_mode: CacheMode = CacheMode.USE_DATASET_USE_FEATURES):
        # download & extract dataset
        dl_cache_dir = os.path.join(DATASET_DOWNLOAD_CACHE, self.args.base_name)
        download_config = DownloadConfig(
            cache_dir=dl_cache_dir,
            force_download=cache_mode == CacheMode.NEW_DATASET_NEW_FEATURES,
        )
        download_manager = DownloadManager(dataset_name=self.args.dataset_name, download_config=download_config)
        self._dl_train_file, self._dl_dev_file = download_manager.download_and_extract(
            [DOWNLOAD_URL_BASE + name for name in [self.train_file_name, self.dev_file_name]]
        )

    def preprocess(self, cache_mode: CacheMode = CacheMode.USE_DATASET_USE_FEATURES):
        # create dataset object
        self._create_squad_training_arguments(
            os.path.dirname(self._dl_train_file), overwrite_cache=cache_mode < CacheMode.USE_DATASET_USE_FEATURES
        )
        self.dataset = {
            self.train_split_name: SquadLikeDataset(
                self.squad_args,
                tokenizer=self.tokenizer,
                cached_features_file=self._get_features_cache_file(self.train_split_name),
                mode="train",
                data_file=os.path.basename(self._dl_train_file),
            ),
            self.dev_split_name: SquadLikeDataset(
                self.squad_args,
                tokenizer=self.tokenizer,
                cached_features_file=self._get_features_cache_file(self.dev_split_name),
                mode="dev",
                data_file=os.path.basename(self._dl_dev_file),
            ),
        }

    def _get_squad_results(self, predictions):
        for start_logits, end_logits, features in zip(predictions[0], predictions[1], self.dev_split.features):
            yield SquadResult(features.unique_id, start_logits, end_logits)

    def compute_metrics(self, predictions, references):
        squad_results = self._get_squad_results(predictions)
        predictions = compute_predictions_logits(
            self.dev_split.examples[:len(predictions[0])],
            self.dev_split.features[:len(predictions[0])],
            squad_results,
            n_best_size=self.squad_args.n_best_size,
            max_answer_length=self.squad_args.max_answer_length,
            do_lower_case=hasattr(self.tokenizer, "do_lower_case") and self.tokenizer.do_lower_case,
            output_prediction_file=None,
            output_nbest_file=None,
            output_null_log_odds_file=None,
            verbose_logging=False,
            version_2_with_negative=self.with_negative,
            null_score_diff_threshold=self.squad_args.null_score_diff_threshold,
            tokenizer=self.tokenizer,
        )
        results = squad_evaluate(self.dev_split.examples[:len(predictions)], predictions)
        return results

    def get_prediction_head_config(self):
        return {
            "head_type": "question_answering",
            "num_labels": 2,
            "layers": 1,
            "activation_function": "tanh",
        }


class SquadV1Manager(QADatasetManager):
    train_file_name = "SQuAD1-1_train.json.gz"
    dev_file_name = "SQuAD1-1_dev.json.gz"


class SquadV2Manager(QADatasetManager):
    with_negative = True
    train_file_name = "SQuAD2-0_train.json.gz"
    dev_file_name = "SQuAD2-0_dev.json.gz"


class DROPManager(QADatasetManager):
    train_file_name = "DROP_train.json.gz"
    dev_file_name = "DROP_dev.json.gz"


class WikiHopManager(QADatasetManager):
    train_file_name = "WikiHop_train.json.gz"
    dev_file_name = "WikiHop_dev.json.gz"


class HotpotQAManager(QADatasetManager):
    train_file_name = "HotpotQA_train.json.gz"
    dev_file_name = "HotpotQA_dev.json.gz"


# class TriviaQAManager(QADatasetManager):
#     train_file_name = "TriviaQA_wiki_train.json.gz"
#     dev_file_name = "TriviaQA_wiki_dev.json.gz"


class ComQAManager(QADatasetManager):
    train_file_name = "ComQA_train.json.gz"
    dev_file_name = "ComQA_dev.json.gz"


class CQManager(QADatasetManager):
    train_file_name = "ComplexQuestions_train.json.gz"
    dev_file_name = "ComplexQuestions_dev.json.gz"


# class CWQManager(QADatasetManager):
#     train_file_name = "ComplexWebQuestions_train.json.gz"
#     dev_file_name = "ComplexWebQuestions_dev.json.gz"


class NewsQAManager(QADatasetManager):
    train_file_name = "NewsQA_train.json.gz"
    dev_file_name = "NewsQA_dev.json.gz"


# class SearchQAManager(QADatasetManager):
#     train_file_name = "SearchQA_train.json.gz"
#     dev_file_name = "SearchQA_dev.json.gz"


class DuoRCParaphraseManager(QADatasetManager):
    with_negative = True
    train_file_name = "DuoRC_Paraphrase_train.json.gz"
    dev_file_name = "DuoRC_Paraphrase_dev.json.gz"


class DuoRCSelfManager(QADatasetManager):
    with_negative = True
    train_file_name = "DuoRC_Self_train.json.gz"
    dev_file_name = "DuoRC_Self_dev.json.gz"


# Not in MultiQA #


class QuorefManager(QADatasetManager):
    def load(self, cache_mode: CacheMode = CacheMode.USE_DATASET_USE_FEATURES):
        # download & extract dataset
        dl_cache_dir = os.path.join(DATASET_DOWNLOAD_CACHE, self.args.base_name)
        download_config = DownloadConfig(
            cache_dir=dl_cache_dir,
            force_download=cache_mode == CacheMode.NEW_DATASET_NEW_FEATURES,
        )
        download_manager = DownloadManager(dataset_name=self.args.dataset_name, download_config=download_config)
        dl_folder_path = download_manager.download_and_extract(
            "https://quoref-dataset.s3-us-west-2.amazonaws.com/train_and_dev/quoref-train-dev-v0.1.zip"
        )
        self._dl_train_file = os.path.join(dl_folder_path, "quoref-train-dev-v0.1", "quoref-train-v0.1.json")
        self._dl_dev_file = os.path.join(dl_folder_path, "quoref-train-dev-v0.1", "quoref-dev-v0.1.json")
