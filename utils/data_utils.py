import logging
import os

import numpy as np
import pandas as pd


def load_data_info_with_split(args):
    """Apply validation split to dataset.

    Args:
        args: Arguments.

    Returns:
        pd.Dataframe: Dataset info.
    """
    dataset_info = pd.read_csv(os.path.join(args.data_path, "dataset.csv"))

    cil_data_info = dataset_info[dataset_info["dataset"] == "CIL"]

    # set splits for CIL dataset
    for idx, row in cil_data_info.iterrows():
        if np.random.rand() <= args.val_split:
            row["split"] = "val"
            cil_data_info.loc[idx] = row

    # set split for DeepGlobe dataset
    if args.datasets == "dg":
        dg_data_info = dataset_info[dataset_info["dataset"] == "DeepGlobe"]
        # set splits for CIL dataset
        for idx, row in dg_data_info.iterrows():
            if np.random.rand() <= args.val_split:
                row["split"] = "val"
                dg_data_info.loc[idx] = row

    if args.datasets == "all":
        dataset_info[dataset_info["dataset"] == "CIL"] = cil_data_info
    elif args.datasets == "cil":
        dataset_info = cil_data_info
    elif args.datasets == "cil-mrd":
        dataset_info[dataset_info["dataset"] == "CIL"] = cil_data_info
        dataset_info = dataset_info[dataset_info["dataset"] != "DeepGlobe"]
    elif args.datasets == "cil-dg":
        dataset_info[dataset_info["dataset"] == "CIL"] = cil_data_info
        dataset_info = dataset_info[dataset_info["dataset"] != "MRD"]
    elif args.datasets == "dg":
        dataset_info = dg_data_info

    # filter out images with few mask pixles
    if args.min_pixels is not None:
        for idx, row in dataset_info.iterrows():
            if row["n_pixels"] < args.min_pixels and row["dataset"] != "CIL":
                row["split"] = "none"
                dataset_info.loc[idx] = row

    train_df = dataset_info[dataset_info["split"] == "train"]
    val_df = dataset_info[dataset_info["split"] == "val"]
    out = dataset_info[dataset_info["split"] == "none"]

    logging.info("Dataset info:")
    logging.info(f"\tTrain dataset: {train_df.shape[0]} images")
    logging.info(f"\tValidation dataset: {val_df.shape[0]} images")
    logging.info(f"\tIgnoring: {out.shape[0]} images")
    logging.info(f"\tUsing Augmentations: {args.augmentation}")
    return train_df, val_df
