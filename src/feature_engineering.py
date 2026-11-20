
def create_features(data):

    # Create a copy so original data remains unchanged
    data = data.copy()

    # 1. Power-to-weight ratio
    data["power_to_weight_ratio"] = (
        data["max_power_bhp"] / data["kerb_weight_kg"]
    )

    # 2. Engine capacity relative to bike weight
    data["engine_cc_per_kg"] = (
        data["engine_cc"] / data["kerb_weight_kg"]
    )

    # 3. Average tyre pressure
    data["average_tyre_pressure"] = (
        data["tyre_pressure_front_psi"]
        + data["tyre_pressure_rear_psi"]
    ) / 2

    # 4. Difference between front and rear tyre pressure
    data["tyre_pressure_difference"] = (
        data["tyre_pressure_front_psi"]
        - data["tyre_pressure_rear_psi"]
    )

    # 5. Difference between maximum and average speed
    data["speed_difference"] = (
        data["max_speed_kmph"]
        - data["avg_speed_kmph"]
    )

    return data
