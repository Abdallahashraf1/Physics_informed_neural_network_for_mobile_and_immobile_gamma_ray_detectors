import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TFSMLayer


# Define the function to load the model
@st.cache_resource
def load_model(model_path):
    model = tf.saved_model.load('my_model.tf')
    return model


# Constants
MASS_ATTENUATION_COEFF = 0.02  # cm^2/g
R_INITIAL = 12.0
ALARM_LEVEL = 1.0  # Constant alarm level

# Define the calculate_immobile_MDD function
def calculate_immobile_MDD(model, angles, params, tolerance=1e-3):
    Pdesired, activity, branching_ratio, air_density, v, acquisition_time, background_count_rate = params
    R_min = 0
    R_max = 100  # Set an upper bound for the distance
    while (R_max - R_min) > tolerance:
        R_fix = (R_min + R_max) / 2
        current_params = (Pdesired, activity, branching_ratio, air_density, v, acquisition_time, background_count_rate, ALARM_LEVEL, R_fix)

        angles_rad = np.deg2rad(angles)
        angles_rad = tf.convert_to_tensor(angles_rad[:, np.newaxis], dtype=tf.float32)

        # Get model predictions
        model_output = model(angles_rad)

        # Ensure model output is a tensor
        if isinstance(model_output, dict) and 'output' in model_output:
            rel_eff = model_output['output']
        elif isinstance(model_output, tf.Tensor):
            rel_eff = model_output
        else:
            raise ValueError("Unexpected format for model output.")

        # Ensure the shape is as expected
        if rel_eff.shape != (angles_rad.shape[0], 1):
            raise ValueError(f"Unexpected shape for rel_eff: {rel_eff.shape}, expected {(angles_rad.shape[0], 1)}")

        fluence_rate = activity * branching_ratio * rel_eff
        detection_probability = 1 - tf.math.exp(-fluence_rate * R_fix * air_density * background_count_rate)

        if tf.reduce_mean(detection_probability) >= Pdesired:
            R_max = R_fix
        else:
            R_min = R_fix

    return R_fix

# Define the calculate_mobile_MDD function
def calculate_mobile_MDD(model, angles, params, tolerance=1e-3):
    Pdesired, activity, branching_ratio, air_density, v, acquisition_time, background_count_rate = params
    R_min = 0
    R_fix = calculate_immobile_MDD(model, angles, params, tolerance=tolerance)  # Starting point for mobile MDD
    R_max = 2 * R_fix  # Initial doubling of R_fix

    while (R_max - R_min) > tolerance:
        R_test = (R_max + R_min) / 2
        current_params = (Pdesired, activity, branching_ratio, air_density, v, acquisition_time, background_count_rate, ALARM_LEVEL, R_test)

        angles_rad = np.deg2rad(angles)
        angles_rad = tf.convert_to_tensor(angles_rad[:, np.newaxis], dtype=tf.float32)

        total_detection_prob = 0.0
        distance_traveled = 0.0
        current_distance_per_step = v * acquisition_time

        while distance_traveled < R_test:
            angle_rad = np.arcsin(distance_traveled / R_test)
            angles_rad = tf.convert_to_tensor([[angle_rad]], dtype=tf.float32)

            # Get model predictions
            model_output = model(angles_rad)

            # Ensure model output is a tensor
            if isinstance(model_output, dict) and 'output' in model_output:
                rel_eff = model_output['output']
            elif isinstance(model_output, tf.Tensor):
                rel_eff = model_output
            else:
                raise ValueError("Unexpected format for model output.")

            # Ensure the shape is as expected
            if rel_eff.shape != (angles_rad.shape[0], 1):
                raise ValueError(f"Unexpected shape for rel_eff: {rel_eff.shape}, expected {(angles_rad.shape[0], 1)}")

            fluence_rate = activity * branching_ratio * rel_eff
            detection_prob = 1 - tf.math.exp(-fluence_rate * R_test * air_density * background_count_rate)
            total_detection_prob += detection_prob
            distance_traveled += current_distance_per_step

        if total_detection_prob >= Pdesired:
            R_max = R_test
        else:
            R_min = R_test

    return R_test

# Define a function to calculate the maximum detectable distance (MDD)
def calculate_mdd(model, angles, params, is_mobile, tolerance):
    if is_mobile:
        return calculate_mobile_MDD(model, angles, params, tolerance)
    else:
        return calculate_immobile_MDD(model, angles, params, tolerance)

# Streamlit UI
st.title("Maximum Detectable Distance of Gamma Rays Detectors")

Pdesired = st.slider("Desired Detection Probability", 0.0, 1.0, 0.95)
angles_input = st.text_input("Angles (comma separated)", "0, 10, 20, 30")

# Input fields for the parameters
col1, col2 = st.columns(2)

with col1:
    background_count_rate = st.number_input("Background Count Rate", value=1.0)
    activity = st.number_input("Activity (uCi)", value=5.2)
    is_mobile = st.checkbox("Is the detector mobile?")

with col2:
    branching_ratio = st.number_input("Branching Ratio", value=0.8519)
    air_density = st.number_input("Air Density (g/cm^2)", value=0.0294)

# Display speed of the detector and acquisition time only if detector is mobile
if is_mobile:
    speed_of_detector = st.number_input("Speed of the Detector/Vehicle (m/s)", value=1.0)
    acquisition_time = st.number_input("Acquisition Time", value=1.0)
else:
    speed_of_detector = 1.0
    acquisition_time = 1.0

tolerance = 1e-3  # Set tolerance for calculation

# Prepare the parameters
angles = [float(angle) for angle in angles_input.split(",")]
params = (Pdesired, activity, branching_ratio, air_density, speed_of_detector, acquisition_time, background_count_rate)

# Load the model
model_path = 'my_model.tf'  # Update this path to your model directory
model = load_model(model_path)

# Calculate MDD
if st.button("Calculate MDD"):
    try:
        print(f"Input angles: {angles}")
        print(f"Input params: {params}")

        MDD = calculate_mdd(model, angles, params, is_mobile, tolerance)

        print(f"MDD calculated: {MDD}")
        st.write(f"Maximum Detectable Distance (MDD): {MDD:.2f} inches")
    except ValueError as ve:
        st.error(str(ve))
