import numpy as np
import pandas as pd

import os
import ray

__author__ = "Nicolas de Montigny"

__all__ = ['load_Xy_data','save_Xy_data']

# Load data from file
def load_Xy_data(Xy_file):
    with np.load(Xy_file, allow_pickle=True) as f:
        return f['data'].tolist()

# Save data to file
def save_Xy_data(df, Xy_file):
    np.savez(Xy_file, data = df)
