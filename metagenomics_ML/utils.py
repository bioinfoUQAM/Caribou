
import pandas as pd
import numpy as np

from sklearn import metrics

__author__ = "nicolas"

# Load data from file
def load_Xy_data(Xy_file):
    if os.path.basename(Xy_file).split(sep = ".")[1] == "hdf5":
        return pd.read_hdf(Xy_file, 'df')
    elif os.path.basename(Xy_file).split(sep = ".")[1] == "npz":
        with np.load(Xy_file, allow_pickle=True) as f:
            return f['data'].tolist()

# Save data to file
def save_Xy_data(data,Xy_file):
    if type(data) == pd.core.frame.DataFrame:
        data.to_hdf(Xy_file, key='df', mode='w', complevel = 9, complib = 'bzip2')
    elif type(data) == dict:
        np.savez(Xy_file, data=data)

# Model training cross-validation stats
def training_cross_validation(y_pred_test, y_test, classifier):
    print("Cross validating classifier : " + str(classifier))

    print("y_pred_test : ")
    print(y_pred_test)
    print("y_test : ")
    print(y_test)

    print("Confidence matrix : ")
    print(str(metrics.confusion_matrix(y_pred_test, y_test)))
    print("Precision : " + str(metrics.precision_score(y_pred_test, y_test)))
    print("Recall : " + str(metrics.recall_score(y_pred_test, y_test)))
    print("F-score : " + str(metrics.f1_score(y_pred_test, y_test)))
    print("AUC ROC : " + str(metrics.roc_auc_score(y_pred_test, y_test)))
