import argparse 
import os
import joblib

import pandas as pd
from sklearn import metrics

import config
import model_dispatcher


def main():
    df = pd.read_csv(config.TRAIN_DATA)


if __name__ == "__main__":
    main()
