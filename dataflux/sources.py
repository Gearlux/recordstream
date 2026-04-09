from typing import Any, Iterator, List, Optional

from confluid import configurable
from logflow import get_logger

from dataflux.sample import Sample

logger = get_logger(__name__)


@configurable
class HuggingFaceSource:
    """
    DataFlux Source for Hugging Face Datasets.
    Configurable mapping of dataset features to DataFlux Sample triplets.
    """

    def __init__(
        self,
        path: str,
        split: str = "train",
        input_feature: str = "image",
        target_feature: str = "label",
        metadata_features: Optional[List[str]] = None,
        count: Optional[int] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        from datasets import load_dataset

        self.path = path
        self.split = split
        self.input_feature = input_feature
        self.target_feature = target_feature
        self.metadata_features = metadata_features or []
        self.count = count

        logger.info(f"HuggingFaceSource: Loading {path} ({split})...")
        self._dataset = load_dataset(path, name=name, split=split, **kwargs)

    def __iter__(self) -> Iterator[Sample]:
        counter = 0
        limit = self.count or len(self._dataset)

        for item in self._dataset:
            if counter >= limit:
                break

            # 1. Extract Input
            input_val = item.get(self.input_feature)

            # 2. Extract Target
            target_val = item.get(self.target_feature)

            # 3. Build Metadata
            metadata = {f: item.get(f) for f in self.metadata_features}
            metadata["hf_path"] = self.path
            metadata["hf_split"] = self.split

            yield Sample(input=input_val, target=target_val, metadata=metadata)
            counter += 1

    def __len__(self) -> int:
        if self.count is not None:
            return self.count
        return len(self._dataset)
