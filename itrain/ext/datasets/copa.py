import os
import urllib.request
from xml.dom import minidom

import datasets
import requests


_CITATION = """\
@inproceedings{roemmele2011choice,
  title={Choice of Plausible Alternatives: An Evaluation of Commonsense Causal Reasoning.},
  author={Roemmele, Melissa and Bejan, Cosmin Adrian and Gordon, Andrew S},
  booktitle={AAAI spring symposium: logical formalizations of commonsense reasoning},
  pages={90--95},
  year={2011}
}
"""

_DESCRIPTION = """\
The Choice Of Plausible Alternatives (COPA) evaluation provides researchers with a tool for assessing progress in open-domain commonsense causal reasoning. COPA consists of 1000 questions, split equally into development and test sets of 500 questions each. Each question is composed of a premise and two alternatives, where the task is to select the alternative that more plausibly has a causal relation with the premise. The correct alternative is randomized so that the expected performance of randomly guessing is 50%.
"""

_DOWNLOAD_URL = "https://people.ict.usc.edu/~gordon/downloads/COPA-resources.tgz"



class COPA(datasets.GeneratorBasedBuilder):
    """The Choice Of Plausible Alternatives (COPA) dataset."""

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "premise": datasets.Value("string"),
                    "alternative1": datasets.Value("string"),
                    "alternative2": datasets.Value("string"),
                    "relation": datasets.Value("string"),
                    "label": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="https://people.ict.usc.edu/~gordon/copa.html",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        def _download(src_url, dst_path):
            # ignore SSL certificate verification during download
            response = requests.get(src_url, stream=True, verify=False)
            with open(dst_path, "wb") as f:
                for data in response.iter_content():
                    f.write(data)
            return dst_path

        dl_dir = dl_manager.extract(dl_manager.download_custom(_DOWNLOAD_URL, _download))
        data_dir = os.path.join(dl_dir, "COPA-resources", "datasets")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": os.path.join(data_dir, "copa-dev.xml")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": os.path.join(data_dir, "copa-test.xml")},
            ),
        ]

    def _generate_examples(self, filepath):
        data = minidom.parse(filepath)
        items = data.getElementsByTagName("item")
        for item in items:
            index = item.attributes["id"].value
            yield index, {
                "id": index,
                "premise": item.getElementsByTagName("p")[0].firstChild.data,
                "alternative1": item.getElementsByTagName("a1")[0].firstChild.data,
                "alternative2": item.getElementsByTagName("a2")[0].firstChild.data,
                "relation": item.attributes["asks-for"].value,
                "label": item.attributes["most-plausible-alternative"].value,
            }
