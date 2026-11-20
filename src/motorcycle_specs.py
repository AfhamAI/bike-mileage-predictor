
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "motorcycle_mileage_dataset.csv"
)


# These are the motorcycle specifications we can retrieve
# automatically from the dataset.
SPECIFICATION_COLUMNS = [
    "segment",
    "engine_cc",
    "max_power_bhp",
    "kerb_weight_kg",
    "fuel_tank_l",
    "transmission",
    "gears",
    "cooling",
    "abs_type",
]


NUMERIC_SPECIFICATIONS = [
    "engine_cc",
    "max_power_bhp",
    "kerb_weight_kg",
    "fuel_tank_l",
    "gears",
]


CATEGORICAL_SPECIFICATIONS = [
    "segment",
    "transmission",
    "cooling",
    "abs_type",
]


def load_motorcycle_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Motorcycle dataset not found: {DATA_PATH}"
        )

    data = pd.read_csv(
        DATA_PATH,
        low_memory=False
    )

    required_columns = [
        "brand",
        "model",
        "model_year",
        *SPECIFICATION_COLUMNS,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing columns: {missing_columns}"
        )

    data = data.copy()

    data["brand"] = (
        data["brand"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    data["model"] = (
        data["model"]
        .astype("string")
        .str.strip()
    )

    data["model_year"] = pd.to_numeric(
        data["model_year"],
        errors="coerce"
    )

    for column in NUMERIC_SPECIFICATIONS:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    return data


# Load the dataset once when this module is imported.
MOTORCYCLE_DATA = load_motorcycle_data()


def get_brands():

    return sorted(
        MOTORCYCLE_DATA["brand"]
        .dropna()
        .unique()
        .tolist()
    )


def get_models(brand):

    data = MOTORCYCLE_DATA[
        MOTORCYCLE_DATA["brand"] == brand
    ]

    return sorted(
        data["model"]
        .dropna()
        .unique()
        .tolist()
    )


def get_model_years(brand, model):

    data = MOTORCYCLE_DATA[
        (MOTORCYCLE_DATA["brand"] == brand)
        & (MOTORCYCLE_DATA["model"] == model)
    ]

    years = (
        data["model_year"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    return sorted(years, reverse=True)


def get_representative_value(data, column):

    values = data[column].dropna()

    if values.empty:
        return None

    if column in NUMERIC_SPECIFICATIONS:

        return float(values.median())

    modes = values.mode()

    if modes.empty:
        return None

    value = modes.iloc[0]

    if pd.isna(value):
        return None

    return str(value)


def get_motorcycle_specs(brand, model, model_year=None):

    matching_data = MOTORCYCLE_DATA[
        (MOTORCYCLE_DATA["brand"] == brand)
        & (MOTORCYCLE_DATA["model"] == model)
    ]

    if matching_data.empty:
        return None

    available_years = get_model_years(brand, model)

    if not available_years:
        return None

    if model_year is None:
        selected_year = available_years[0]

    else:
        try:
            selected_year = int(model_year)
        except (ValueError, TypeError):
            return None

        if selected_year not in available_years:
            return None

    year_data = matching_data[
        matching_data["model_year"] == selected_year
    ]

    if year_data.empty:
        return None

    # Use the selected model year's records.
    # If a specification is missing for that year,
    # fall back to records for the same motorcycle model.
    specifications = {}

    for column in SPECIFICATION_COLUMNS:

        value = get_representative_value(
            year_data,
            column
        )

        if value is None:
            value = get_representative_value(
                matching_data,
                column
            )

        specifications[column] = value

    specifications["model_year"] = selected_year

    return specifications
