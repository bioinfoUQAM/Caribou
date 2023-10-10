from collections import Counter

import pandas as pd

from ray.data.preprocessor import Preprocessor

TENSOR_COLUMN_NAME = '__value__'

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
        df.loc[:,TENSOR_COLUMN_NAME] = pd.Series(list(tensors))
        df = df.drop(columns = [self.column])
        return df
    
    def __repr__(self):
        fn_name = getattr(self.tokenization_fn, "__name__", self.tokenization_fn)
        return (
            f"{self.__class__.__name__}(column = {self.column!r}, tokenization_fn = {fn_name})"
        )


