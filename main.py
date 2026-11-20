from src.data_loader import load_raw_data
from src.data_cleaning import clean_data
from config import PROCESSED_DATA_PATH
from src.feature_engineering import create_features


def main():

    # Load original dataset
    data = load_raw_data()


    # Clean dataset
    cleaned_data = clean_data(data)

    # Create processed folder if it doesn't exist
    PROCESSED_DATA_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save cleaned dataset
    output_path = (PROCESSED_DATA_PATH / "motorcycle_mileage_cleaned.csv")
    cleaned_data.to_csv(output_path, index=False)


    # Create new features
    featured_data = create_features(cleaned_data)

    print("Original columns:", cleaned_data.shape[1])
    print("New columns:", featured_data.shape[1])

    print("\nNew Features:")
    print(
        featured_data[
            [
                "power_to_weight_ratio",
                "engine_cc_per_kg",
                "average_tyre_pressure",
                "tyre_pressure_difference",
                "speed_difference"
            ]
        ].head()
    )


if __name__ == "__main__":
    main()
