import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# -------------------------------
# Page Setup
# -------------------------------
st.set_page_config(
    page_title="Acoustic Noise & Rocket Efficiency Model",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Acoustic Noise Impact on Rocket Fuel Consumption and Efficiency")
st.write(
    "This dashboard analyzes how acoustic noise affects rocket fuel consumption "
    "and engine efficiency using regression-based modeling."
)

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file is not None:

    df = pd.read_excel(uploaded_file)

    st.subheader("1. Dataset Preview")
    st.dataframe(df.head())

    st.write("Columns found in your Excel file:")
    st.write(list(df.columns))

    # -------------------------------
    # Column Selection
    # -------------------------------
    st.subheader("2. Select Variables")

    noise_col = st.selectbox(
        "Select Acoustic Noise column (Independent Variable)",
        df.columns
    )

    fuel_col = st.selectbox(
        "Select Fuel Consumption column (Dependent Variable 1)",
        df.columns
    )

    efficiency_col = st.selectbox(
        "Select Efficiency column (Dependent Variable 2)",
        df.columns
    )

    # -------------------------------
    # Convert columns to numeric safely
    # -------------------------------
    if noise_col not in df.columns or fuel_col not in df.columns or efficiency_col not in df.columns:
        st.error("One or more selected columns are invalid.")
        st.stop()

    try:
        # Use .iloc[:,0] to get a Series if needed
        df[noise_col] = df[[noise_col]].astype(float)
        df[fuel_col] = df[[fuel_col]].astype(float)
        df[efficiency_col] = df[[efficiency_col]].astype(float)
    except Exception as e:
        st.error(f"Failed to convert columns to numeric: {e}")
        st.stop()

    # Drop any rows with missing values
    data = df[[noise_col, fuel_col, efficiency_col]].dropna()

    if data.empty:
        st.error("No valid numeric data available in selected columns.")
        st.stop()

    X = data[[noise_col]]
    y_fuel = data[fuel_col]
    y_eff = data[efficiency_col]

    # -------------------------------
    # Regression Model: Fuel Consumption
    # -------------------------------
    fuel_model = LinearRegression()
    fuel_model.fit(X, y_fuel)
    fuel_pred = fuel_model.predict(X)

    fuel_slope = float(np.ravel(fuel_model.coef_)[0])
    fuel_intercept = float(np.ravel([fuel_model.intercept_])[0])
    fuel_r2 = float(r2_score(y_fuel, fuel_pred))

    # -------------------------------
    # Regression Model: Efficiency
    # -------------------------------
    eff_model = LinearRegression()
    eff_model.fit(X, y_eff)
    eff_pred = eff_model.predict(X)

    eff_slope = float(np.ravel(eff_model.coef_)[0])
    eff_intercept = float(np.ravel([eff_model.intercept_])[0])
    eff_r2 = float(r2_score(y_eff, eff_pred))

    # -------------------------------
    # Display Regression Results
    # -------------------------------
    st.subheader("3. Regression Results")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Acoustic Noise vs Fuel Consumption")
        st.write(f"**Equation:** Fuel Consumption = {fuel_intercept:.2f} + ({fuel_slope:.2f} × Acoustic Noise)")
        st.write(f"**R² Value:** {fuel_r2:.4f}")
        st.info("Interpretation: Fuel consumption increases as acoustic noise increases.")

    with col2:
        st.markdown("### Acoustic Noise vs Efficiency")
        st.write(f"**Equation:** Efficiency = {eff_intercept:.2f} + ({eff_slope:.4f} × Acoustic Noise)")
        st.write(f"**R² Value:** {eff_r2:.4f}")
        st.info("Interpretation: Efficiency decreases as acoustic noise increases.")

    # -------------------------------
    # Prediction Section
    # -------------------------------
    st.subheader("4. Prediction Model")

    min_noise = float(data[noise_col].min())
    max_noise = float(data[noise_col].max())
    mean_noise = float(data[noise_col].mean())

    noise_input = st.slider(
        "Enter Acoustic Noise Level (dB)",
        min_value=min_noise,
        max_value=max_noise,
        value=mean_noise
    )

    input_df = pd.DataFrame({noise_col: [noise_input]})
    predicted_fuel = float(np.ravel(fuel_model.predict(input_df))[0])
    predicted_eff = float(np.ravel(eff_model.predict(input_df))[0])

    col3, col4 = st.columns(2)
    with col3:
        st.metric("Predicted Fuel Consumption", f"{predicted_fuel:.2f}")
    with col4:
        st.metric("Predicted Efficiency", f"{predicted_eff:.2f}%")

    # -------------------------------
    # Instability Zone
    # -------------------------------
    st.subheader("5. Acoustic Instability Zone")

    if noise_input < 140:
        zone = "Stable Zone"
        decision = "Low acoustic disturbance. Combustion stable."
        recommendation = "No major control required."
    elif 140 <= noise_input < 150:
        zone = "Transition Zone"
        decision = "Early impact on fuel and efficiency."
        recommendation = "Monitor pressure oscillations."
    elif 150 <= noise_input <= 155:
        zone = "Critical Instability Zone"
        decision = "Significant increase in fuel consumption and drop in efficiency."
        recommendation = "Use resonators/baffles to reduce instability."
    elif 155 < noise_input <= 170:
        zone = "Unstable Zone"
        decision = "Strong acoustic instability affecting performance."
        recommendation = "Improve chamber/nozzle design."
    else:
        zone = "Severe Instability Zone"
        decision = "Severe fuel loss and efficiency drop."
        recommendation = "Immediate acoustic noise control required."

    st.warning(f"Current Zone: {zone}")
    st.write(f"**Decision:** {decision}")
    st.write(f"**Recommendation:** {recommendation}")

    # -------------------------------
    # Final Notes
    # -------------------------------
    st.subheader("6. Notes")
    st.write("This model is based on secondary/simulated data and theoretical analysis. Experimental validation is not included.")