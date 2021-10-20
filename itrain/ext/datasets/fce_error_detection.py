import logging
import os

import datasets


_CITATION = """\
@inproceedings{rei-yannakoudakis-2016-compositional,
    title = "Compositional Sequence Labeling Models for Error Detection in Learner Writing",
    author = "Rei, Marek  and
      Yannakoudakis, Helen",
    booktitle = "Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2016",
    address = "Berlin, Germany",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P16-1112",
    doi = "10.18653/v1/P16-1112",
    pages = "1181--1191",
}
"""

_DESCRIPTION = """\
This is a version of the publicly released FCE dataset (Yannakoudakis et al., 2011), modified for experiments on error detection. Rei & Yannakoudakis (2016) describe the creation of this version, and report the first error detection results using the corpus. If you're using this dataset, please reference these two papers.

The original dataset contains 1141 examination scripts for training, 97 for testing, and 6 for outlier experiments. We randomly separated 80 training scripts as a development set, use the same test set, and do not make use of the outlier scripts.

The texts were first tokenised and sentence split by RASP (Briscoe et al., 2006) and then each token was assigned a binary label. Tokens that have been annotated within an error tag are labeled as incorrect (i), otherwise they are labeled as correct (c). When dealing with missing words or phrases, the error label is assigned to the next token in the sequence. FCE has american spelling marked as erroneous (error type "SA"), which we ignore in this dataset and regard as correct.
"""

_DOWNLOAD_URL = "https://s3-eu-west-1.amazonaws.com/ilexir-website-media/fce-error-detection.tar.gz"


class FCEErrorDetection(datasets.GeneratorBasedBuilder):
    """FCE dataset for error detection."""

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "labels": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "c",
                                "i",
                            ]
                        )
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://www.ilexir.co.uk/datasets/index.html",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        dl_dir = dl_manager.download_and_extract(_DOWNLOAD_URL)
        data_dir = os.path.join(dl_dir, "fce-error-detection", "tsv")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": os.path.join(data_dir, "fce-public.train.original.tsv")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": os.path.join(data_dir, "fce-public.dev.original.tsv")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": os.path.join(data_dir, "fce-public.test.original.tsv")},
            ),
        ]

    def _generate_examples(self, filepath):
        logging.info(f"Generating examples from {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            index = 0
            tokens = []
            labels = []
            for line in f:
                if line == "\n":
                    # example complete
                    yield index, {
                        "tokens": tokens,
                        "labels": labels,
                    }
                    index += 1
                    tokens = []
                    labels = []
                else:
                    token, label = line.split("\t")
                    tokens.append(token.strip())
                    labels.append(label.strip())
            # last
            yield index, {
                "tokens": tokens,
                "labels": labels,
            }
