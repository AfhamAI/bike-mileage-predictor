
import pandas as pd


def clean_data(data):

    # Create a copy so the original dataset remains unchanged
    data = data.copy()

    # 1. Remove duplicate rows
    data = data.drop_duplicates()

    # 2. Clean brand names
    data["brand"] = data["brand"].str.strip().str.upper()

    # 3. Standardize road types
    data["road_type"] = data["road_type"].str.strip().str.lower()

    road_mapping = {
        "city": "city_street",
        "city street": "city_street",
        "hwy": "highway",
        "pothole": "potholed",
        "rural": "rural_road",
        "highway": "highway",
        "expressway": "expressway",
        "gravel": "gravel",
    }

    data["road_type"] = data["road_type"].replace(road_mapping)

    # 4. Standardize weather
    data["weather"] = data["weather"].str.strip().str.lower()

    weather_mapping = {
        "sunny": "clear",
        "rain": "rainy",
        "downpour": "heavy_rain",
        "heavy rain": "heavy_rain",
        "cool": "cold",
        "fog": "foggy",
    }

    data["weather"] = data["weather"].replace(weather_mapping)

    # 5. Convert engine_cc into numeric values
    data["engine_cc"] = (
        data["engine_cc"]
        .astype(str)
        .str.lower()
        .str.replace("cc", "", regex=False)
        .str.strip()
    )

    data["engine_cc"] = pd.to_numeric(
        data["engine_cc"],
        errors="coerce"
    )

    # 6. Replace invalid rider weights with missing values
    data.loc[
        data["rider_weight_kg"].isin([0, 5, 450]),
        "rider_weight_kg"
    ] = None

    # 7. Return cleaned dataset
    return data
