
from pathlib import Path
import math

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

from src.motorcycle_specs import (
    get_brands,
    get_models,
    get_model_years,
    get_motorcycle_specs,
)



# 1. CONFIGURATION


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "artifacts"
    / "final_mileage_pipeline.joblib"
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)


 
# 2. EXACT 34 MODEL FEATURES
 

FEATURE_COLUMNS = [
    "brand",
    "model",
    "segment",
    "engine_cc",
    "max_power_bhp",
    "kerb_weight_kg",
    "fuel_tank_l",
    "transmission",
    "gears",
    "cooling",
    "abs_type",
    "model_year",
    "odometer_km",
    "months_since_service",
    "tyre_type",
    "tyre_condition",
    "tyre_pressure_front_psi",
    "tyre_pressure_rear_psi",
    "time_of_day",
    "road_type",
    "road_surface",
    "surface_condition",
    "weather",
    "traffic_density",
    "stops_per_km",
    "gradient_pct",
    "avg_speed_kmph",
    "max_speed_kmph",
    "riding_style",
    "rider_weight_kg",
    "pillion",
    "luggage_kg",
    "fuel_type",
    "trip_distance_km",
]


# Fail early if the saved pipeline expects a different schema.
model_features = getattr(model, "feature_names_in_", None)

if model_features is not None:
    if list(model_features) != FEATURE_COLUMNS:
        raise ValueError(
            "The saved model's input features do not match "
            "the 34 features configured in app.py."
        )


 
# 3. MOTORCYCLE SPECIFICATIONS
 

BRANDS = get_brands()


 
# 4. USER INPUT OPTIONS
 

OPTIONS = {

    "tyre_type": [
        "tube",
        "tubeless",
    ],

    "tyre_condition": [
        "average",
        "good",
        "new",
        "worn",
    ],

    "time_of_day": [
        "morning",
        "afternoon",
        "evening",
        "night",
    ],

    "road_type": [
        "city_street",
        "expressway",
        "flooded",
        "ghat",
        "gravel",
        "highway",
        "hill",
        "hill_ghat",
        "mud",
        "potholed",
        "rural_road",
        "village_mud",
        "waterlogged",
    ],

    "road_surface": [
        "asphalt",
        "concrete",
        "dirt",
        "gravel",
    ],

    "surface_condition": [
        "damp",
        "dry",
        "flooded",
        "icy",
        "wet",
    ],

    "weather": [
        "clear",
        "cold",
        "foggy",
        "heavy_rain",
        "hot",
        "humid",
        "rainy",
        "windy",
    ],

    "traffic_density": [
        "heavy",
        "light",
        "moderate",
    ],

    "riding_style": [
        "aggressive",
        "eco",
        "normal",
    ],

    "fuel_type": [
        "Premium",
        "Regular_E0",
        "Regular_E10",
        "Regular_E20",
    ],
}


 
# 5. USER INPUT DEFINITIONS
 

FIELD_CONFIG = {

    "time_of_day": {
        "label": "Time of day",
        "type": "select",
        "default": "morning",
        "required": True,
    },

    "road_type": {
        "label": "Road type",
        "type": "select",
        "default": "city_street",
        "required": True,
    },

    "weather": {
        "label": "Weather",
        "type": "select",
        "default": "clear",
        "required": True,
    },

    "traffic_density": {
        "label": "Traffic density",
        "type": "select",
        "default": "moderate",
        "required": True,
    },

    "avg_speed_kmph": {
        "label": "Expected average speed (KM/H)",
        "type": "number",
        "default": 40,
        "required": True,
        "step": "any",
    },

    "riding_style": {
        "label": "Riding style",
        "type": "select",
        "default": "normal",
        "required": True,
    },

    "rider_weight_kg": {
        "label": "Rider weight (KG)",
        "type": "number",
        "default": 65,
        "required": True,
        "step": "any",
    },

    "pillion": {
        "label": "Pillion passenger",
        "type": "select",
        "default": 0,
        "required": True,
        "options": {
            "0": "No",
            "1": "Yes",
        },
    },

    "luggage_kg": {
        "label": "Luggage weight (KG)",
        "type": "number",
        "default": 0,
        "required": True,
        "step": "any",
    },

    "fuel_type": {
        "label": "Fuel type",
        "type": "select",
        "default": "Regular_E10",
        "required": True,
    },

    "trip_distance_km": {
        "label": "Trip distance (KM)",
        "type": "number",
        "default": 20,
        "required": True,
        "step": "any",
    },

    "months_since_service": {
        "label": "Months since last service",
        "type": "number",
        "default": 3,
        "required": True,
        "step": "1",
    },

    "tyre_condition": {
        "label": "Tyre condition",
        "type": "select",
        "default": "good",
        "required": True,
    },

    "road_surface": {
        "label": "Road surface",
        "type": "select",
        "default": "asphalt",
        "required": False,
    },

    "surface_condition": {
        "label": "Surface condition",
        "type": "select",
        "default": "dry",
        "required": False,
    },

    "stops_per_km": {
        "label": "Stops per KM",
        "type": "number",
        "default": 0.2,
        "required": False,
        "step": "any",
    },

    "gradient_pct": {
        "label": "Road gradient (%)",
        "type": "number",
        "default": 0,
        "required": False,
        "step": "any",
    },

    "max_speed_kmph": {
        "label": "Maximum speed (KM/H)",
        "type": "number",
        "default": 80,
        "required": False,
        "step": "any",
    },

    "odometer_km": {
        "label": "Odometer reading (KM)",
        "type": "number",
        "default": 5000,
        "required": False,
        "step": "any",
    },

    "tyre_type": {
        "label": "Tyre type",
        "type": "select",
        "default": "tubeless",
        "required": False,
    },

    "tyre_pressure_front_psi": {
        "label": "Front tyre pressure (PSI)",
        "type": "number",
        "default": 29,
        "required": False,
        "step": "any",
    },

    "tyre_pressure_rear_psi": {
        "label": "Rear tyre pressure (PSI)",
        "type": "number",
        "default": 33,
        "required": False,
        "step": "any",
    },
}


 
# 6. FORM SECTIONS
 

FORM_SECTIONS = [

    {
        "title": "Trip Details",
        "description": "Tell us about the journey you want to estimate.",
        "fields": [
            "time_of_day",
            "road_type",
            "weather",
            "traffic_density",
            "avg_speed_kmph",
            "riding_style",
            "trip_distance_km",
        ],
    },

    {
        "title": "Rider & Fuel",
        "description": "Enter rider, passenger and fuel information.",
        "fields": [
            "rider_weight_kg",
            "pillion",
            "luggage_kg",
            "fuel_type",
        ],
    },

    {
        "title": "Maintenance",
        "description": "These details help account for motorcycle condition.",
        "fields": [
            "months_since_service",
            "tyre_condition",
        ],
    },

    {
        "title": "Advanced Trip & Maintenance Details",
        "description": (
            "Optional. Default values are used if you leave these unchanged."
        ),
        "fields": [
            "road_surface",
            "surface_condition",
            "stops_per_km",
            "gradient_pct",
            "max_speed_kmph",
            "odometer_km",
            "tyre_type",
            "tyre_pressure_front_psi",
            "tyre_pressure_rear_psi",
        ],
    },
]


 
# 7. BUILD FORM DATA
 

def build_form_fields(values):

    fields = {}

    for name, config in FIELD_CONFIG.items():

        options = config.get(
            "options",
            OPTIONS.get(name)
        )

        fields[name] = {
            "name": name,
            "label": config["label"],
            "type": config["type"],
            "value": values.get(
                name,
                config["default"]
            ),
            "default": config["default"],
            "required": config["required"],
            "options": options,
            "step": config.get("step", "any"),
        }

    return fields


def build_form_sections(fields):

    sections = []

    for section in FORM_SECTIONS:

        sections.append({
            "title": section["title"],
            "description": section["description"],
            "fields": [
                fields[name]
                for name in section["fields"]
            ],
            "optional": (
                section["title"]
                == "Advanced Trip & Maintenance Details"
            ),
        })

    return sections


 
# 8. BUILD COMPLETE MODEL INPUT
 

def build_model_input(brand, motorcycle, year, form_values):

    specifications = get_motorcycle_specs(
        brand,
        motorcycle,
        year
    )

    if specifications is None:
        raise ValueError(
            "Could not find specifications for this motorcycle. "
            "Please select another model or model year."
        )

    # Start with automatically retrieved motorcycle specifications.
    input_data = {
        "brand": brand,
        "model": motorcycle,
        **specifications,
    }

    # Add user-provided and default trip/maintenance information.
    for field, config in FIELD_CONFIG.items():

        raw_value = form_values.get(
            field,
            config["default"]
        )

        if raw_value is None or str(raw_value).strip() == "":
            raw_value = config["default"]

        if config["type"] == "number":

            try:
                value = float(raw_value)
            except (ValueError, TypeError):
                raise ValueError(
                    f"Please enter a valid number for "
                    f"{config['label']}."
                )

            if not math.isfinite(value):
                raise ValueError(
                    f"Invalid value for {config['label']}."
                )

            input_data[field] = value

        else:

            valid_options = config.get(
                "options",
                OPTIONS.get(field, [])
            )

            if isinstance(valid_options, dict):
                valid_values = list(valid_options.keys())
            else:
                valid_values = valid_options

            if str(raw_value) not in [
                str(option) for option in valid_values
            ]:
                raise ValueError(
                    f"Invalid selection for {config['label']}."
                )

            if field == "pillion":
                input_data[field] = int(raw_value)
            else:
                input_data[field] = str(raw_value)

    # Ensure the model receives all 34 features in the exact order.
    input_df = pd.DataFrame(
        [input_data],
        columns=FEATURE_COLUMNS
    )

    return input_df



# 9. VALIDATE COMPLETE INPUT


def validate_model_input(input_df):

    row = input_df.iloc[0]

    positive_fields = [
        "engine_cc",
        "max_power_bhp",
        "kerb_weight_kg",
        "fuel_tank_l",
        "gears",
        "avg_speed_kmph",
        "rider_weight_kg",
        "trip_distance_km",
    ]

    for field in positive_fields:

        if row[field] <= 0:
            raise ValueError(
                f"{field.replace('_', ' ').title()} "
                "must be greater than zero."
            )

    non_negative_fields = [
        "odometer_km",
        "months_since_service",
        "tyre_pressure_front_psi",
        "tyre_pressure_rear_psi",
        "stops_per_km",
        "luggage_kg",
        "max_speed_kmph",
    ]

    for field in non_negative_fields:

        if row[field] < 0:
            raise ValueError(
                f"{field.replace('_', ' ').title()} "
                "cannot be negative."
            )

    if row["max_speed_kmph"] < row["avg_speed_kmph"]:
        raise ValueError(
            "Maximum speed cannot be lower than average speed."
        )

    if row["pillion"] not in [0, 1]:
        raise ValueError(
            "Pillion must be either Yes or No."
        )

    if row["model_year"] <= 0:
        raise ValueError("Invalid motorcycle model year.")



# 10. MOTORCYCLE DATA API


@app.route("/api/models")
def api_models():

    brand = request.args.get("brand", "").strip().upper()

    if brand not in BRANDS:
        return jsonify({
            "models": [],
            "error": "Invalid brand."
        }), 400

    return jsonify({
        "models": get_models(brand)
    })


@app.route("/api/motorcycle-specs")
def api_motorcycle_specs():

    brand = request.args.get("brand", "").strip().upper()
    motorcycle = request.args.get("model", "").strip()

    if brand not in BRANDS:
        return jsonify({
            "error": "Invalid brand."
        }), 400

    if motorcycle not in get_models(brand):
        return jsonify({
            "error": "Invalid motorcycle model."
        }), 400

    years = get_model_years(brand, motorcycle)

    if not years:
        return jsonify({
            "error": "No model years found."
        }), 404

    requested_year = request.args.get("year")

    if requested_year:
        try:
            selected_year = int(requested_year)
        except ValueError:
            return jsonify({
                "error": "Invalid model year."
            }), 400
    else:
        selected_year = years[0]

    specifications = get_motorcycle_specs(
        brand,
        motorcycle,
        selected_year
    )

    if specifications is None:
        return jsonify({
            "error": "Specifications not found."
        }), 404

    return jsonify({
        "years": years,
        "selected_year": selected_year,
        "specifications": specifications,
    })



# 11. MAIN PREDICTION ROUTE


@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    error = None
    input_summary = None

    values = {
        name: config["default"]
        for name, config in FIELD_CONFIG.items()
    }

    selected_brand = BRANDS[0] if BRANDS else ""
    selected_model = ""
    selected_year = None

    if request.method == "POST":

        selected_brand = request.form.get(
            "brand", ""
        ).strip().upper()

        selected_model = request.form.get(
            "model", ""
        ).strip()

        year_value = request.form.get(
            "model_year", ""
        ).strip()

        for field in FIELD_CONFIG:

            values[field] = request.form.get(
                field,
                FIELD_CONFIG[field]["default"]
            )

        try:

            if selected_brand not in BRANDS:
                raise ValueError("Please select a valid brand.")

            if selected_model not in get_models(selected_brand):
                raise ValueError(
                    "Please select a valid model for this brand."
                )

            if not year_value:
                raise ValueError(
                    "Please select the motorcycle model year."
                )

            selected_year = int(year_value)

            input_df = build_model_input(
                selected_brand,
                selected_model,
                selected_year,
                values
            )

            validate_model_input(input_df)

            # Predict using the saved complete pipeline.
            result = model.predict(input_df)[0]

            if not math.isfinite(float(result)):
                raise ValueError(
                    "The model returned an invalid prediction."
                )

            prediction = round(float(result), 2)

            input_summary = {
                "brand": selected_brand,
                "model": selected_model,
                "year": selected_year,
                "trip_distance": input_df.iloc[0]["trip_distance_km"],
            }

        except (ValueError, TypeError, OverflowError) as exception:

            error = str(exception)

        except Exception:

            app.logger.exception("Prediction failed.")

            error = (
                "Prediction failed. Check the Flask terminal "
                "for the technical error."
            )

    # Initial values for the motorcycle dropdown.
    available_models = (
        get_models(selected_brand)
        if selected_brand in BRANDS
        else []
    )

    if selected_model not in available_models:
        selected_model = (
            available_models[0]
            if available_models
            else ""
        )

    available_years = (
        get_model_years(selected_brand, selected_model)
        if selected_model
        else []
    )

    if selected_year not in available_years:
        selected_year = (
            available_years[0]
            if available_years
            else None
        )

    fields = build_form_fields(values)

    sections = build_form_sections(fields)

    return render_template(
        "index.html",
        brands=BRANDS,
        models=available_models,
        years=available_years,
        selected_brand=selected_brand,
        selected_model=selected_model,
        selected_year=selected_year,
        sections=sections,
        prediction=prediction,
        error=error,
        input_summary=input_summary,
    )



# 12. RUN FLASK


if __name__ == "__main__":

    print("\n" + "=" * 55)
    print("MOTORCYCLE MILEAGE PREDICTOR")
    print("=" * 55)

    print("Saved ML pipeline loaded successfully.")
    print("Expected model features:", len(FEATURE_COLUMNS))
    print("Available brands:", len(BRANDS))

    print("\nStarting Flask server...")

    app.run(
        host="127.0.0.1",
        port=8000,
        debug=True
    )
