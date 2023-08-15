from typing import List
from collections import Counter

import pandas as pd

from ray.data import Dataset
from ray.data.preprocessor import Preprocessor

from ray.data.preprocessors import CountVectorizer

class KmersVectorizer(CountVectorizer):
    """
    Class adapted from ray.data.preprocessors.CountVectorizer to debug a pandas warning and better adapt to K-mers
    Computes all the k-mers that can be found in the sequences in the order they were seen then removes 25% most & least common
    Takes in one column and a k value
    """
    def __init__(
        self,
        k,
        column: str
    ):
        def kmer_tokenize(s):
            tokens = []
            for start in range(0, len(s)-k, 1):
                token = s[start:start+k]
                if 'N' not in token:
                    tokens.append(token)
            return tokens
        self.column = column
        
        return super().__init__(
            columns = [self.column],
            tokenization_fn = kmer_tokenize
        )
    
    def _fit(self, dataset: Dataset) -> Preprocessor:
        def get_pd_value_counts(df: pd.DataFrame) -> List[Counter]:
            def get_token_counts(col):
                token_series = df[col].apply(self.tokenization_fn)
                tokens = token_series.sum()
                return Counter(tokens)

            return [get_token_counts(col) for col in self.columns]

        value_counts = dataset.map_batches(
            get_pd_value_counts,
            batch_format="pandas"
            # batch_size = 1
        )
        total_counts = [Counter() for _ in self.columns]
        # for batch in value_counts.iter_batches(batch_size=1):
        for batch in value_counts.iter_batches(batch_size=None):
            for i, col_value_counts in enumerate(batch):
                total_counts[i].update(col_value_counts)
        
        def most_common(counter: Counter, n: int):
            return Counter(dict(counter.most_common(n)))
        
        def least_common(counter: Counter, n: int):
            return Counter(dict(counter.most_common()[:-n-1:-1]))

        nb_quantile = int(len(total_counts[0])/4)

        top_counts = most_common(total_counts[0], nb_quantile)

        bottom_counts = least_common(total_counts[0], nb_quantile)

        mid_counts = Counter({
            token : count for token, count in total_counts[0].items() if token not in top_counts.keys() and token not in bottom_counts.keys()
        })

        self.stats_ = {
            f"token_counts({self.column})": mid_counts
        }

        return self


    def _transform_pandas(self, df: pd.DataFrame):
        mapping = {
            'id' : df['id']
        }
        token_counts = self.stats_[f"token_counts({self.column})"]
        tokenized = df[self.column].map(self.tokenization_fn).map(Counter)
        for token in token_counts:
            mapping[token] = tokenized.map(lambda val: val[token])
        df = pd.concat(mapping, axis = 1)
        return df