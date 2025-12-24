import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from joblib import load, dump
import os
import warnings
from sklearn.utils import resample
import copy

warnings.filterwarnings("ignore")


def incremental_train_from_csv(
        base_model_path,
        update_csv_path,
        output_model_path,
        reference_train_csv,
        weight_new_data=5.0,  # 新增参数：新数据的权重倍数
):
    if not os.path.exists(base_model_path):
        print("pre_train model not exist")
        return False

    if not os.path.exists(update_csv_path):
        print("update_csv not exist")
        return False

    # load pre_train model
    model = load(base_model_path)

    ref_data = pd.read_csv(reference_train_csv, index_col=0)
    ref_data["SB_update_count"] = ref_data["SB_update_count"].apply(
        lambda x: 1 if x > 0 else 0
    )

    X_ref = ref_data.drop("SB_update_count", axis=1).values
    scaler = MinMaxScaler()
    scaler.fit(X_ref)

    update_data = pd.read_csv(update_csv_path, index_col=0)

    if "SB_update_count" not in update_data.columns:
        print("update_csv lack SB_update_count")
        return False

    update_data["SB_update_count"] = update_data["SB_update_count"].apply(
        lambda x: 1 if x > 0 else 0
    )

    X_new = update_data.drop("SB_update_count", axis=1).values
    y_new = update_data["SB_update_count"].values

    if len(X_new) == 0:
        print("no data in update_csv")
        return False

    X_new = scaler.transform(X_new)

    # increase sample weight
    X_train = X_new
    y_train = y_new
    sample_weights = np.ones(len(X_new)) * weight_new_data


    if sample_weights is not None:
        model.fit(X_train, y_train, sample_weight=sample_weights)
        print(f"finish incremental learning (weight={weight_new_data})")

        dump(model, output_model_path)
        print(f"new model save to: {output_model_path}")

        return True


# example
def main():
    success = incremental_train_from_csv(
        base_model_path="./model/model.joblib",
        update_csv_path="./data/merge_csv/warm_cold1616.csv",
        output_model_path="./model/model_incremental_weighted.joblib",
        reference_train_csv="./data/merge_csv/warm_cold1616.csv",
    )

    if success:
        return 0
    else:
        return 1


if __name__ == "__main__":
    main()