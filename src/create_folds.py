import pandas as pd
import config
from sklearn import model_selection

if __name__ == "__main__":
    print("Reading data!")
    df = pd.read_csv(config.TRANSFORMED_DATA)
    df["kfold"] = -1

    print("Data loaded!")
    df = df.sample(frac=1).reset_index(drop=True)

    kf = model_selection.StratifiedKFold(
        n_splits=5, shuffle=False, random_state=config.SEED
    )

    print("Starting loop!")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=df.target.values)):
        print(len(train_idx), len(val_idx))
        df.loc[val_idx, "kfold"] = fold

    print("Saving data!")
    df.to_csv(config.TRAIN_DATA, index=False)
