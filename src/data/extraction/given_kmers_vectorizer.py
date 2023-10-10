from typing import List

from data.extraction.kmers_vectorizer import KmersVectorizer

class GivenKmersVectorizer(KmersVectorizer):

    _is_fittable = False

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