import pandas as pd
from pathlib import Path

# data_dir = Path("../dataSet/AudioWAV")

BASE_DIR = Path(__file__).resolve().parent.parent
data_dir = BASE_DIR / "dataSet" / "AudioWAV"

emotion_map = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad"
}

sentence_map = {
    "IEO": "It's eleven o'clock.",
    "TIE": "That is exactly what happened.",
    "IOM": "I'm on my way to the meeting.",
    "IWW": "I wonder what this is about.",
    "TAI": "The airplane is almost full.",
    "MTI": "Maybe tomorrow it will be cold.",
    "IWL": "I would like a new alarm clock.",
    "ITH": "I think I have a doctor's appointment.",
    "DFA": "Don't forget a jacket.",
    "ITS": "I think I've seen this before.",
    "TSI": "The surface is slick.",
    "WSI": "We'll stop in a couple of minutes."
}

intensity_map = {
    "LO": "Low intensity",
    "MD": "Medium intensity",
    "HI": "High intensity",
    "XX": "Unspecified intensity"
    }


def getRecords(dir):
    records = []
    for file in dir.glob("*.wav"):
        parts = file.stem.split("_")
        if len(parts)!=4:
            continue
        actor_id, sentence, emotion_code, intensity = parts
        records.append({
            "file_path": str(file),
            "actor_id": int(actor_id),
            "sentence_code": sentence,
            "sentence": sentence_map.get(sentence,pd.NA),
            "emotion_code": emotion_code,
            "emotion": emotion_map.get(emotion_code, pd.NA),
            "intensity_code": intensity,
            "intensity": intensity_map.get(intensity,pd.NA)
        })
    return records

def getDataSet_1():
    df = pd.DataFrame(getRecords(dir=data_dir))
    return df

df = getDataSet_1()

print(df.shape)

