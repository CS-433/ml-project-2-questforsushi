from simpletransformers.classification import ClassificationModel
import csv
import pandas as pd
import torch
import numpy as np


TEST_DATA_PATH = "Tweet_Large_Files/twitter-datasets/test_data.txt"
MODEL_DATA_PATH = "./Tweet_Large_Files/checkpoint-10000-epoch-1"
PRED_CSV_PATH = "aicrowd_csv/"
CSV_NAME = "vinai_100k.csv"


# Check if cuda (GPU) is available
cuda = torch.cuda.is_available()
if cuda:
  print("Cuda available - Uses GPU")
else:
  print("Cuda unavailable - Uses CPU")

# we import test data
test_data = pd.read_csv(TEST_DATA_PATH, names = ["text"], sep='\n', header = None, dtype='str', quoting = csv.QUOTE_NONE)

# we import the trained model
vinai = ClassificationModel("bertweet", MODEL_DATA_PATH, use_cuda=cuda)

# we run the predictions
preds, _ = vinai.predict(test_data.to_numpy().tolist())

# we set up the right indices and convert the [0,1] predictions to [-1,1]
df_preds = pd.DataFrame(preds, columns=["Prediction"])
df_preds.index += 1
df_preds.index.rename("Id", inplace=True)
df_preds[df_preds["Prediction"] == 0] = -1

# we export the data in a csv
df_preds.to_csv(PRED_CSV_PATH+CSV_NAME)