from collections import defaultdict
from typing import Generic, Iterator, List, Optional, Sequence, Sized, TypeVar

import torch
from torch.utils.data import Sampler


class RandomSampler(Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    Ported from https://pytorch.org/docs/stable/_modules/torch/utils/data/sampler.html#RandomSampler
    with adaptation to allow subset without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """
    data_source: Sized
    replacement: bool

    def __init__(
        self, data_source: Sized, replacement: bool = False, num_samples: Optional[int] = None, generator=None
    ) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self.generator = generator

        if not isinstance(num_samples, int):
            raise ValueError(f"num_samples should be an integer but got num_samples={self.num_samples}")

        if num_samples <= 0 or num_samples > len(self.data_source):
            self._num_samples = None
        else:
            self._num_samples = num_samples

    @property
    def num_samples(self) -> int:
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=self.generator).tolist()
            yield from torch.randint(
                high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=self.generator
            ).tolist()
        else:
            yield from torch.randperm(n, generator=self.generator).tolist()[: self.num_samples]

    def __len__(self):
        return self.num_samples


class StratifiedRandomSampler(Sampler):
    r"""Samples a stratified subset from a sequence of class labels.
    Adapted from https://pytorch.org/docs/stable/_modules/torch/utils/data/sampler.html#WeightedRandomSampler.

    Args:
        weights (sequence)   : a sequence of class labels
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.
        generator (Generator): Generator used in sampling.
    """
    weights: torch.Tensor
    num_samples: int
    replacement: bool

    def __init__(self, class_labels: Sequence, num_samples: int, replacement: bool = False, generator=None) -> None:
        if not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer " "value, but got num_samples={}".format(num_samples)
            )

        # count occurences
        class_counts = defaultdict(int)
        for example in class_labels:
            class_counts[example] += 1
        total = float(len(class_labels))
        # set weights
        weights = []
        for example in class_labels:
            weights.append(class_counts[example] / total)
        self.weights = torch.as_tensor(weights, dtype=torch.double)

        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator

    def __iter__(self):
        rand_tensor = torch.multinomial(self.weights, self.num_samples, self.replacement, generator=self.generator)
        return iter(rand_tensor.tolist())

    def __len__(self):
        return self.num_samples


class QAPossibleSubsetRandomSampler(Sampler):
    """Samples a random subset from the possible examples of a SQuAD-style QA dataset."""

    num_samples: int
    replacement: bool

    def __init__(self, features: Sequence, num_samples: int, replacement: bool = False, generator=None) -> None:
        if not isinstance(num_samples, int):
            raise ValueError(
                "num_samples should be an integer " "value, but got num_samples={}".format(num_samples)
            )
        if num_samples <= 0:
            num_samples = len(features)

        # count possible instances
        possible = 0
        weights = []
        for feature in features:
            if not feature.is_impossible:
                possible += 1
                weights.append(1)
            else:
                weights.append(0)
        self.weights = torch.as_tensor(weights, dtype=torch.double) * (possible / len(features))

        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator

    def __iter__(self):
        rand_tensor = torch.multinomial(self.weights, self.num_samples, self.replacement, generator=self.generator)
        return iter(rand_tensor.tolist())

    def __len__(self):
        return self.num_samples
