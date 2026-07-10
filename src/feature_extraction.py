import pandas as pd
import opensmile 
from tqdm import tqdm
from dataset import getDataSet_1


df = getDataSet_1()
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals
)

feature_row = []


def feature_extraction():
    print("Feature Extraction Start")
    for _, row in tqdm(df.iterrows(),total=len(df),desc="Extraction Features"):
        features = smile.process_file(row["file_path"])
        features["emotion"] = row["emotion"]
        features["actor_id"] = row["actor_id"]
        features["sentence"] = row["sentence"]
        features["intensity"] = row["intensity"]
        feature_row.append(features)
    features_df = pd.concat(feature_row,ignore_index=True)
    return features_df

features_df = feature_extraction()
features_df.to_csv(
    "egemaps_features.csv",
    index=False
)
print(features_df.head(5))