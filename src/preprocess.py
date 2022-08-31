import pandas as pd
import numpy as np
import config


def main():
    df = pd.read_feather(config.RAW_DATA)
    print("Loaded data!")

    list_of_cols = [x for x in df.columns if x not in ["customer_ID", "S_2", "target"]]
    num_features = [x for x in list_of_cols if x not in config.CAT_FEATURES]
    cat_features = config.CAT_FEATURES.copy()
    num_features = [col for col in list_of_cols if col not in cat_features]

    print("Aggregating features!")
    test_num_agg = df.groupby("customer_ID")[num_features].agg(
        ["mean", "std", "min", "max", "last"]
    )
    test_num_agg.columns = ["_".join(x) for x in test_num_agg.columns]

    test_cat_agg = df.groupby("customer_ID")[cat_features].agg(
        ["count", "last", "nunique"]
    )
    test_cat_agg.columns = ["_".join(x) for x in test_cat_agg.columns]

    print("Merging new features!")
    df = pd.concat([test_num_agg, test_cat_agg], axis=1)

    print("Loading labels-data!")
    df_labels = pd.read_csv(config.TRAINING_LABELS)
    df_labels = df_labels.set_index("customer_ID")

    print("Merging features with target!")
    df_train = df.merge(df_labels, left_index=True, right_index=True, how="left")
    df_train.target = df_train.target.astype("int8")
    df_train = df_train.sort_index().reset_index()

    print("Saving data!")
    df.to_csv(config.TRANSFORMED_DATA)


if __name__ == "__main__":
    main()
