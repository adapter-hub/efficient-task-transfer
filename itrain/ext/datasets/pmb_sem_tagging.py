import logging
import os

import datasets


_CITATION = """\
@inproceedings{AbzianidzeBos2017IWCS,
   title     = {Towards Universal Semantic Tagging},
   author    = {Abzianidze, Lasha and Bos, Johan},
   booktitle = {Proceedings of the 12th International Conference on Computational Semantics (IWCS 2017) -- Short Papers},
   address   = {Montpellier, France},
   pages     = {1--6},
   year      = {2017}
}
"""

_DESCRIPTION = """\
The universal semantic tagging is a sequence labeling task where labels are language-neutral, semantically informative tags.

This release is a frozen snapshot of a subset of English PMB documents that are marked as gold or silver standard in the current development version.
The gold folder contains tagged documents that are fully manually checked, while the silver contains tagged documents that are only partially manually checked.
WARNING: use silver documents at your own risk!
"""

_DOWNLOAD_URL = "https://pmb.let.rug.nl/releases/sem-0.1.0.zip"


class PMBSemanticTagging(datasets.GeneratorBasedBuilder):
    """English documents form the Parallel Meaning Bank tagged with the universal semantic tags"""

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "sem_tags": datasets.Sequence(datasets.Value("string")),
                }
            ),
            supervised_keys=None,
            homepage="https://pmb.let.rug.nl/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        dl_dir = dl_manager.download_and_extract(_DOWNLOAD_URL)
        base_data_dir = os.path.join(dl_dir, "sem-0.1.0", "data")

        return [
            datasets.SplitGenerator(
                name="silver",
                gen_kwargs={"data_dir": os.path.join(base_data_dir, "silver")},
            ),
            datasets.SplitGenerator(
                name="gold",
                gen_kwargs={"data_dir": os.path.join(base_data_dir, "gold")},
            ),
        ]

    def _generate_examples(self, data_dir):
        logging.info(f"Generating examples from {data_dir}")
        for filename in os.listdir(data_dir):
            filepath = os.path.join(data_dir, filename)
            tokens = []
            sem_tags = []
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    tag, token = line.split("\t")
                    tokens.append(token.strip())
                    sem_tags.append(tag.strip())
            yield filename, {
                "id": filename,
                "tokens": tokens,
                "sem_tags": sem_tags,
            }
