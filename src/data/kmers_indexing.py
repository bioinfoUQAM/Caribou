
class kmers_index():
    # k-mers indexing
    """
    - https://www.geeksforgeeks.org/create-inverted-index-for-file-using-python/
    - https://www.coursera.org/lecture/dna-sequencing/practical-implementing-a-k-mer-index-o5wzo
    
    UDF for indexing in dictionnary (hash map) list of kmers extracted with KMC
    1. Extract k-mers using KMC
    2. Extract total list of k-mers found -> dict.keys
    3. Iterate over sequences to populate count vectors / key (k-mer)
    4. Ray dataset from items -> vector count matrix

    """
    # Count vectorizing
    """
    1. https://docs.ray.io/en/latest/ray-air/api/doc/ray.data.preprocessors.Tokenizer.html#ray.data.preprocessors.Tokenizer
    THEN
        2.1 https://docs.ray.io/en/latest/ray-air/api/doc/ray.data.preprocessors.CountVectorizer.html#ray.data.preprocessors.CountVectorizer
    OR
        2.2 https://docs.ray.io/en/latest/ray-air/api/doc/ray.data.preprocessors.HashingVectorizer.html#ray.data.preprocessors.HashingVectorizer

    UDF tokenization function to use extracted kmers with KMC -> Count vectorization ?
    2. UDF to split sequence into k-mers found
    3. Overload CountVectorizer Class from Ray to use index
        3.1 Fit : Build index
        3.2 Transform : Extend into 1 column / k-mer
            3.2.1 Iterate over sequences to populate index
            3.2.2 Add one column per k-mer found in the index
            3.2.3 Populate each column using the index
    
    """
    # Sub-word tokenization
    """
    https://www.tensorflow.org/text/guide/subwords_tokenizer
    https://huggingface.co/docs/transformers/tokenizer_summary#subword-tokenization
    https://colab.research.google.com/github/tensorflow/text/blob/master/docs/guide/subwords_tokenizer.ipynb
    https://ai.googleblog.com/2021/12/a-fast-wordpiece-tokenization-system.html
    https://www.nltk.org/api/nltk.tokenize.html
    https://blog.octanove.org/guide-to-subword-tokenization/
    https://towardsdatascience.com/a-comprehensive-guide-to-subword-tokenisers-4bbd3bad9a7c
    https://towardsdatascience.com/word-subword-and-character-based-tokenization-know-the-difference-ea0976b64e17
    https://towardsdatascience.com/byte-pair-encoding-subword-based-tokenization-algorithm-77828a70bee0
    https://towardsdatascience.com/wordpiece-subword-based-tokenization-algorithm-1fbd14394ed7
    """
    def __init__(self):
        super(self.kmers_index)

    def index_sequence(self):
        print('to_do')

    def query_sequence(self):
        print('todo')