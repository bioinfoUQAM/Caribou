from typing import List
from itertools import compress
from collections import Counter

import pandas as pd

from ray.data import Dataset
from ray.data.preprocessor import Preprocessor

class KmersVectorizer(Preprocessor):
    """
    Class adapted from ray.data.preprocessors.CountVectorizer to debug a pandas warning and better adapt to K-mers
    Computes all the k-mers that can be found in the sequences in the order they were seen and keeps only the ones that are represented by ATCG
    """
    def __init__(
        self,
        k,
        column: str
    ):
        def kmer_tokenize(s):
            tokens = []
            for start in range(0, len(s)-k, 1):
                tokens.append(s[start:start+k])
            return tokens
        self.column = column
        self.tokenization_fn = kmer_tokenize

    def _transform_pandas(self, df: pd.DataFrame):
        mapping = {}
        tokens = self.stats_[f"tokens({self.column})"]
        tokenized = df[self.column].map(self.tokenization_fn).map(Counter)
        for token in tokens:
            mapping[token] = tokenized.map(lambda val: val[token])
        mapping = pd.concat(mapping, axis = 1)
        tensors = mapping.to_numpy()
        df.loc[:,'__value__'] = pd.Series(list(tensors))
        df = df.drop(columns = [self.column])
        return df
    
    def __repr__(self):
        fn_name = getattr(self.tokenization_fn, "__name__", self.tokenization_fn)
        return (
            f"{self.__class__.__name__}(column = {self.column!r}, tokenization_fn = {fn_name})"
        )

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

class GivenKmersVectorizer(KmersVectorizer):

    def __init__(
        self,
        k,
        column: str,
        tokens: List[str]
    ):
        super().__init__(
            k,
            column
        )
        self.stats_ = {
            f"tokens({self.column})": tokens
        }

    def _fit(self, dataset: Dataset) -> Preprocessor:
        return self
