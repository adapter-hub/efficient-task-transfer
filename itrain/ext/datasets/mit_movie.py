import logging

import datasets


_DESCRIPTION = """\
The MIT Movie Corpus is a semantically tagged training and test corpus in BIO format. The eng corpus are simple queries, and the trivia10k13 corpus are more complex queries.
"""

_DOWNLOAD_URL = "https://groups.csail.mit.edu/sls/downloads/movie/"


class MITMovieConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class MITMovie(datasets.GeneratorBasedBuilder):
    """MIT Movie Corpus for NER."""

    BUILDER_CONFIGS = [
        MITMovieConfig(name="eng", version=datasets.Version("1.0.0"), description="MIT Movie simple queries"),
        MITMovieConfig(name="trivia", version=datasets.Version("1.0.0"), description="MIT Movie complex queries"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(datasets.Value("string")),
                }
            ),
            supervised_keys=None,
            homepage="https://groups.csail.mit.edu/sls/downloads/",
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        if self.config.name == "trivia":
            dl_name = "trivia10k13"
        else:
            dl_name = "eng"
        dl_urls = {
            "train": f"{_DOWNLOAD_URL}{dl_name}train.bio",
            "test": f"{_DOWNLOAD_URL}{dl_name}test.bio",
        }
        dl_files = dl_manager.download_and_extract(dl_urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": dl_files["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": dl_files["test"]},
            ),
        ]

    def _generate_examples(self, filepath):
        logging.info(f"Generating examples from {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            index = 0
            tokens = []
            tags = []
            for line in f:
                if line == "\n":
                    # example complete
                    yield index, {
                        "tokens": tokens,
                        "ner_tags": tags,
                    }
                    index += 1
                    tokens = []
                    tags = []
                else:
                    tag, token = line.split("\t")
                    tokens.append(token.strip())
                    tags.append(tag.strip())
            # last
            yield index, {
                "tokens": tokens,
                "ner_tags": tags,
            }
