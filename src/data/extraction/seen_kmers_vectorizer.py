from typing import List
from collections import Counter

import pandas as pd

from ray.data import Dataset
from ray.data.preprocessor import Preprocessor
from data.extraction.kmers_vectorizer import KmersVectorizer

class SeenKmersVectorizer(KmersVectorizer):

    def __init__(
        self,
        k,
        column: str
    ):
        super().__init__(
            k,
            column
        )
        
    def _fit(self, dataset: Dataset) -> Preprocessor:
        def get_pd_value_counts(df: pd.DataFrame) -> List[Counter]:
            def get_token_counts(col):
                token_series = df[col].apply(self.tokenization_fn)
                tokens = token_series.sum()
                return Counter(tokens)

            return {self.column : [get_token_counts(self.column)]}

        value_counts = dataset.map_batches(
            get_pd_value_counts,
            batch_format = "pandas"
        )

        total_counts = Counter()
        for batch in value_counts.iter_batches(batch_size=None):
            for col_value_counts in batch[self.column]:
                total_counts.update(col_value_counts)

        alphabet = set('ATCG')
        tokens = [token for token in total_counts if set(token) <= alphabet]
        self.stats_ = {
            f"tokens({self.column})": tokens
        }

        return self