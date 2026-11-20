from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent

# Dataset paths
RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "motorcycle_mileage_dataset.csv"
)

PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"

# Model settings
TARGET_COLUMN = "mileage_kmpl"

ID_COLUMNS = ["record_id"]

RANDOM_STATE = 42

# Project settings
PROJECT_NAME = "Motorcycle Mileage Prediction"
