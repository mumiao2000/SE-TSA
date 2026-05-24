import argparse
import json
import os
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def _load_preprocessed_dataset(processed_dataset_path: str) -> pd.DataFrame:
    return pd.read_csv(processed_dataset_path, index_col=0)


def _load_raw_dataset(raw_dataset_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw_dataset_path)
    # Process date columns and set index
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        if "Date" in df.columns:
            df = df.drop(columns=["Date"])
    elif "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    else:
        raise ValueError(f"No 'date' or 'Date' column found in {raw_dataset_path}")
    df = df.sort_index()
    # Convert date columns to numeric timestamps
    for col in ["start_date", "end_date", "Month"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].astype(np.int64) / 1e9 / (3600 * 24)
    # Reorder target column to first position if it exists
    target_col = "OT"
    if target_col in df.columns:
        cols = [target_col] + [c for c in df.columns if c != target_col]
        df = df[cols]
    # Handle missing values
    df = df.dropna(axis=1, how="all")
    df = df.ffill().fillna(0)
    return df


def _compute_stats(
    input_len: int,
    output_len: int,
    io_pairs: list,
    train_set: np.ndarray,
    target_col: str,
    column_names: list[str],
    output_dir: str,
) -> Dict[str, Any]:
    n_samples, n_features = train_set.shape
    means = train_set.mean(axis=0).tolist()
    stds = train_set.std(axis=0).tolist()
    mins = train_set.min(axis=0).tolist()
    maxs = train_set.max(axis=0).tolist()
    time_idx = np.arange(n_samples)
    trend_strengths = []
    for i in range(n_features):
        col = train_set[:, i]
        if np.std(col) == 0:
            trend_strengths.append(0.0)
        else:
            corr = np.corrcoef(time_idx, col)[0, 1]
            trend_strengths.append(float(corr))
    target_idx = column_names.index(target_col) if target_col in column_names else 0
    if n_samples > 1:
        fft_vals = np.fft.rfft(train_set[:, target_idx])
        freqs = np.fft.rfftfreq(n_samples)
        dominant_period = float(
            1.0 / (freqs[np.argmax(np.abs(fft_vals[1:])) + 1] + 1e-10)
        )
    else:
        dominant_period = 0.0
    stats = {
        "input_output_info": {
            "input_len": input_len,
            "output_len": output_len,
            "expected_input_shape": f"(batch_size, {input_len}, {n_features})",
            "expected_output_shape": f"(batch_size, {output_len}, {n_features})",
            "io_pairs": io_pairs,
        },
        "train_set_stats": {
            "n_samples": n_samples,
            "n_features": n_features,
            "feature_means": means,
            "feature_stds": stds,
            "feature_mins": mins,
            "feature_maxs": maxs,
            "trend_strengths": trend_strengths,
            "dominant_period": dominant_period,
            "column_names": column_names,
        },
    }
    # save to output path with formatting for readability
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)
    return stats


def _create_input_output_pairs(
    data: np.ndarray, input_len: int, output_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for i in range(len(data) - input_len - output_len + 1):
        x.append(data[i : i + input_len])
        y.append(data[i + input_len : i + input_len + output_len])
    return np.array(x), np.array(y)


def load_dataset(
    args: argparse.Namespace,
) -> Tuple[
    Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]], Dict[str, Any]
]:
    dataset = args.dataset
    raw_dataset_path = f"datasets/raw/{dataset}.csv"
    processed_dataset_path = f"datasets/processed/{dataset}.csv"
    # Preprocess dataset
    if os.path.exists(processed_dataset_path):
        print(f"[Dataset] Loading processed dataset from {processed_dataset_path}")
        df = _load_preprocessed_dataset(processed_dataset_path)
    else:
        print(f"[Dataset] Loading raw dataset from {raw_dataset_path}")
        df = _load_raw_dataset(raw_dataset_path)
        # Save processed dataset for future use
        os.makedirs(os.path.dirname(processed_dataset_path), exist_ok=True)
        df.to_csv(processed_dataset_path)
    # Prepare train/val/test splits (once, chronologically) on DataFrames
    n = len(df)
    n_train = int(n * args.split_ratio[0])
    n_val = int(n * args.split_ratio[1])
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train : n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val :].copy()
    # Process text columns: fit SVD on train set, apply to val/test
    text_cols = [c for c in df.columns if df[c].dtype == object or df[c].dtype == "str"]
    if text_cols:
        model = SentenceTransformer(args.bert_model_name)
        for col in text_cols:
            texts = df[col].fillna("").astype(str).tolist()
            embeddings = model.encode(texts, show_progress_bar=False)
            train_emb = embeddings[:n_train]
            val_emb = embeddings[n_train : n_train + n_val]
            test_emb = embeddings[n_train + n_val :]
            # Fit SVD on train embeddings only
            _, _, Vh = np.linalg.svd(train_emb, full_matrices=False)
            proj = Vh[: args.svd_n_components, :].T
            train_reduced = (train_emb @ proj).astype(np.float32)
            val_reduced = (val_emb @ proj).astype(np.float32)
            test_reduced = (test_emb @ proj).astype(np.float32)
            train_svd = pd.DataFrame(
                train_reduced,
                columns=[f"{col}_svd_{i}" for i in range(args.svd_n_components)],
                index=train_df.index,
            )
            val_svd = pd.DataFrame(
                val_reduced,
                columns=[f"{col}_svd_{i}" for i in range(args.svd_n_components)],
                index=val_df.index,
            )
            test_svd = pd.DataFrame(
                test_reduced,
                columns=[f"{col}_svd_{i}" for i in range(args.svd_n_components)],
                index=test_df.index,
            )
            train_df = pd.concat([train_df.drop(columns=[col]), train_svd], axis=1)
            val_df = pd.concat([val_df.drop(columns=[col]), val_svd], axis=1)
            test_df = pd.concat([test_df.drop(columns=[col]), test_svd], axis=1)
    # Convert to numpy arrays
    train_arr = train_df.values.astype(np.float32)
    val_arr = val_df.values.astype(np.float32)
    test_arr = test_df.values.astype(np.float32)
    # Compute statistics on training set (using first io pair for reference)
    stats = _compute_stats(
        input_len=args.input_len,
        output_len=args.output_len,
        io_pairs=args.io_pairs,
        train_set=train_arr,
        target_col=args.target_col,
        column_names=train_df.columns.tolist(),
        output_dir=args.output_dir,
    )
    print(f"[Dataset] Stats\n{json.dumps(stats, indent=4, ensure_ascii=False)}")
    # Generate input-output pairs for each condition
    dataset_dict = {}
    for il, ol in args.io_pairs:
        train_xy = _create_input_output_pairs(train_arr, il, ol)
        val_xy = _create_input_output_pairs(val_arr, il, ol)
        test_xy = _create_input_output_pairs(test_arr, il, ol)
        dataset_dict[(il, ol)] = (train_xy, val_xy, test_xy)
    return dataset_dict, stats
