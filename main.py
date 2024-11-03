from data_preparator.data_preparator import DataPreparator

# from train_model import TrainModel
from model.model_wrapper import ModelWrapper

from config import PREPARED_DATA_PATH
from config import RAW_DATA_DICT
from config import FEATURE_IMPORTANCES_PATH
from config import SUBMISSION_PATH

# Server
import pandas as pd
from fastapi import FastAPI

app = FastAPI(
    title="Probability default model.",
    description="A simple pipeline to train and use probability default models.",
    version="1.0",
)


# Read and prepare data
@app.post("/prepare_data")
def prepare_data():
    data_preparator = DataPreparator(RAW_DATA_DICT)
    print("Data preparator was initialized.")
    prepared_df = data_preparator.prepare_data()
    prepared_df.to_parquet(PREPARED_DATA_PATH, index=False)


@app.post("/train")
def train_model():
    # read prepared data
    df = pd.read_parquet(PREPARED_DATA_PATH)
    num_folds = 5

    # model_trainer = TrainModel()
    model_wrapper = ModelWrapper(num_folds=num_folds, df=df)
    # feature_importances_df = model_trainer.train_model()
    feature_importances_df = model_wrapper.train_model()
    feature_importances_df.to_csv(FEATURE_IMPORTANCES_PATH, index=False)


@app.post("/predict")
def predict():
    model_wrapper = ModelWrapper()
    submission = model_wrapper.predict()
    submission.to_csv(SUBMISSION_PATH, index=False)
