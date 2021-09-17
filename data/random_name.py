from Bio import SeqIO
import string
import random

original_file = "/home/nicolas/github/metagenomics_ML/data/metagenome/mock/HMP_MOCK.fasta"
corrected_file = "/home/nicolas/github/metagenomics_ML/data/metagenome/mock/mock.fasta"

letters = string.ascii_lowercase

with open(original_file) as original, open(corrected_file, 'a') as corrected:
    records = SeqIO.parse(original_file, 'fasta')
    for record in records:
        new_name = ''.join(random.choice(letters) for i in range(10))
        record.id = new_name
        record.name = new_name
        record.description = new_name
        SeqIO.write(record, corrected, 'fasta')
