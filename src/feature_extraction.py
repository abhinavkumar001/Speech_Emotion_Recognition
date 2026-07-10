import pandas as pd
from pathlib import Path
import opensmile
from tqdm import tqdm

from dataset import getDataSet_1

OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_CSV = OUTPUT_DIR / "egemaps_features.csv"
FAILED_CSV = OUTPUT_DIR / "failed_files.csv"
CHUNK_SIZE = 50

META_COLS = [
    "file_path",
    "actor_id",
    "sentence_code",
    "sentence",
    "emotion_code",
    "emotion",
    "intensity_code",
    "intensity",
]

df = getDataSet_1()

# Skip already processed files
if OUTPUT_CSV.exists():
    processed_files = set(
        pd.read_csv(
            OUTPUT_CSV,
            usecols=["file_path"]
        )["file_path"]
    )

    df = df[~df["file_path"].isin(processed_files)]
    print(f"Skipping {len(processed_files)} files.")
else:
    processed_files = set()

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
    num_workers=4,
)


def process_chunk(chunk_df):
    rows = []
    failed = []

    for _, row in chunk_df.iterrows():
        try:
            feats = smile.process_file(
                row["file_path"]
            ).reset_index(drop=True)

            for col in META_COLS:
                feats[col] = row[col]

            rows.append(feats)

        except Exception:
            failed.append(row["file_path"])

    if rows:
        return pd.concat(rows, ignore_index=True), failed

    return pd.DataFrame(), failed


def save_chunk(chunk_result):
    if chunk_result.empty:
        return

    header = not OUTPUT_CSV.exists()

    chunk_result.to_csv(
        OUTPUT_CSV,
        mode="a",
        header=header,
        index=False
    )


def feature_extraction():
    if len(df) == 0:
        print("All files already processed.")
        return

    chunks = [
        df.iloc[i:i + CHUNK_SIZE]
        for i in range(0, len(df), CHUNK_SIZE)
    ]

    failed_files = []

    for chunk in tqdm(
        chunks,
        desc="Extracting Features"
    ):
        result, failed = process_chunk(chunk)

        save_chunk(result)

        failed_files.extend(failed)

    if failed_files:
        pd.Series(
            failed_files,
            name="file_path"
        ).to_csv(
            FAILED_CSV,
            index=False
        )

        print(f"{len(failed_files)} files failed.")

    print("Feature extraction completed.")


if __name__ == "__main__":
    feature_extraction()