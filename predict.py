
from pathlib import Path

import joblib
import pandas as pd

from src.motorcycle_specs import (
    get_brands,
    get_models,
    get_model_years,
    get_motorcycle_specs,
)


  # 1. LOAD TRAINED MODEL
  
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "artifacts"
    / "final_mileage_pipeline.joblib"
)

model = joblib.load(MODEL_PATH)

print("\n====================================")
print(" MOTORCYCLE MILEAGE PREDICTOR")
print("====================================")


  # 2. SELECT MOTORCYCLE
  
brands = get_brands()

print("\nAvailable Brands:")

for i, brand in enumerate(brands, start=1):
    print(f"{i}. {brand}")

brand_choice = int(input("\nSelect brand number: "))

brand = brands[brand_choice - 1]


models = get_models(brand)

print("\nAvailable Models:")

for i, motorcycle in enumerate(models, start=1):
    print(f"{i}. {motorcycle}")

model_choice = int(input("\nSelect model number: "))

motorcycle = models[model_choice - 1]


years = get_model_years(brand, motorcycle)

print("\nAvailable Years:")

for i, year in enumerate(years, start=1):
    print(f"{i}. {year}")

year_choice = int(input("\nSelect year number: "))

year = years[year_choice - 1]


  # 3. GET MOTORCYCLE SPECIFICATIONS
  
specifications = get_motorcycle_specs(
    brand,
    motorcycle,
    year
)

if specifications is None:
    raise ValueError("Motorcycle specifications not found.")

print("\nMotorcycle specifications loaded successfully.")


  # 4. GET USER INPUT
  
print("\n========== TRIP DETAILS ==========")

time_of_day = input("Time of day (morning/afternoon/evening/night): ")
road_type = input("Road type (city_street/highway/ghat): ")
weather = input("Weather (clear/rainy/foggy): ")
traffic_density = input("Traffic (light/moderate/heavy): ")

avg_speed_kmph = float(input("Average speed (KM/H): "))
max_speed_kmph = float(input("Maximum speed (KM/H): "))

riding_style = input("Riding style (eco/normal/aggressive): ")

trip_distance_km = float(input("Trip distance (KM): "))

print("\n========== RIDER DETAILS ==========")

rider_weight_kg = float(input("Rider weight (KG): "))

pillion = int(input("Pillion? (0 = No, 1 = Yes): "))

luggage_kg = float(input("Luggage weight (KG): "))

fuel_type = input(
    "Fuel type (Premium/Regular_E0/Regular_E10/Regular_E20): "
)

print("\n========== MAINTENANCE ==========")

odometer_km = float(input("Odometer reading (KM): "))

months_since_service = float(
    input("Months since last service: ")
)

tyre_type = input("Tyre type (tube/tubeless): ")

tyre_condition = input(
    "Tyre condition (new/good/average/worn): "
)

tyre_pressure_front_psi = float(
    input("Front tyre pressure (PSI): ")
)

tyre_pressure_rear_psi = float(
    input("Rear tyre pressure (PSI): ")
)

print("\n========== ADVANCED DETAILS ==========")

road_surface = input("Road surface (asphalt/concrete/dirt/gravel): ")

surface_condition = input(
    "Surface condition (dry/wet/damp/flooded/icy): "
)

stops_per_km = float(input("Stops per KM: "))

gradient_pct = float(input("Road gradient (%): "))


  # 5. COMBINE ALL 34 FEATURES
  
input_data = {

    "brand": brand,
    "model": motorcycle,

    **specifications,

    "model_year": year,

    "odometer_km": odometer_km,
    "months_since_service": months_since_service,

    "tyre_type": tyre_type,
    "tyre_condition": tyre_condition,

    "tyre_pressure_front_psi": tyre_pressure_front_psi,
    "tyre_pressure_rear_psi": tyre_pressure_rear_psi,

    "time_of_day": time_of_day,
    "road_type": road_type,
    "road_surface": road_surface,
    "surface_condition": surface_condition,

    "weather": weather,
    "traffic_density": traffic_density,

    "stops_per_km": stops_per_km,
    "gradient_pct": gradient_pct,

    "avg_speed_kmph": avg_speed_kmph,
    "max_speed_kmph": max_speed_kmph,

    "riding_style": riding_style,

    "rider_weight_kg": rider_weight_kg,
    "pillion": pillion,
    "luggage_kg": luggage_kg,

    "fuel_type": fuel_type,

    "trip_distance_km": trip_distance_km,
}


  # 6. CREATE DATAFRAME

FEATURE_COLUMNS = list(model.feature_names_in_)

input_df = pd.DataFrame(
    [input_data],
    columns=FEATURE_COLUMNS
)


  # 7. PREDICT MILEAGE
  
prediction = model.predict(input_df)[0]


  # 8. DISPLAY RESULT
  
print("\n====================================")
print("       PREDICTION RESULT")
print("====================================")

print(f"Brand          : {brand}")
print(f"Motorcycle     : {motorcycle}")
print(f"Model Year     : {year}")
print(f"Trip Distance  : {trip_distance_km} KM")

print("------------------------------------")

print(f"Predicted Mileage: {prediction:.2f} KMPL")

print("====================================")
