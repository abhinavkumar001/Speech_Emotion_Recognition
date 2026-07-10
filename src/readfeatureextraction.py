from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

csv_path = Path(__file__).resolve().parent / "egemaps_features.csv"

df = pd.read_csv(csv_path)
print(df.head())
print(df.columns)



# "F0semitoneFrom27.5Hz_sma3nz_amean"
# "spectralFluxUV_sma3nz_amean"
# "alphaRatioUV_sma3nz_amean"

df.boxplot(
    column="loudness_sma3_amean",
    by="emotion",
    figsize=(8, 5)
)

plt.show()