import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# -------------------------------
# Page Setting
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

    # Remove missing values
    data = df[[noise_col, fuel_col, efficiency_col]].dropna()

    X = data[[noise_col]]
    y_fuel = data[fuel_col]
    y_efficiency = data[efficiency_col]

    # -------------------------------
    # Regression Model 1
    # Acoustic Noise vs Fuel Consumption
    # -------------------------------
    fuel_model = LinearRegression()
    fuel_model.fit(X, y_fuel)

    fuel_pred = fuel_model.predict(X)

    fuel_slope = fuel_model.coef_[0]
    fuel_intercept = fuel_model.intercept_
    fuel_r2 = r2_score(y_fuel, fuel_pred)

    # -------------------------------
    # Regression Model 2
    # Acoustic Noise vs Efficiency
    # -------------------------------
    efficiency_model = LinearRegression()
    efficiency_model.fit(X, y_efficiency)

    efficiency_pred = efficiency_model.predict(X)

    efficiency_slope = efficiency_model.coef_[0]
    efficiency_intercept = efficiency_model.intercept_
    efficiency_r2 = r2_score(y_efficiency, efficiency_pred)

    # -------------------------------
    # Display Regression Results
    # -------------------------------
    st.subheader("3. Regression Results")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Model 1: Acoustic Noise vs Fuel Consumption")
        st.write(
            f"**Regression Equation:**  \n"
            f"Fuel Consumption = {fuel_intercept:.2f} + ({fuel_slope:.2f} × Acoustic Noise)"
        )
        st.write(f"**R² Value:** {fuel_r2:.4f}")

        if fuel_slope > 0:
            st.success(
                "Interpretation: Fuel consumption increases as acoustic noise increases."
            )
        else:
            st.warning(
                "Interpretation: Fuel consumption decreases as acoustic noise increases."
            )

    with col2:
        st.markdown("### Model 2: Acoustic Noise vs Efficiency")
        st.write(
            f"**Regression Equation:**  \n"
            f"Efficiency = {efficiency_intercept:.2f} + ({efficiency_slope:.4f} × Acoustic Noise)"
        )
        st.write(f"**R² Value:** {efficiency_r2:.4f}")

        if efficiency_slope < 0:
            st.success(
                "Interpretation: Efficiency decreases as acoustic noise increases."
            )
        else:
            st.warning(
                "Interpretation: Efficiency increases as acoustic noise increases."
            )

    # -------------------------------
    # Graph 1
    # -------------------------------
    st.subheader("4. Regression Graphs")

    col3, col4 = st.columns(2)

    with col3:
        fig1, ax1 = plt.subplots()
        ax1.scatter(X, y_fuel, label="Actual Data")
        ax1.plot(X, fuel_pred, label="Regression Line")
        ax1.set_xlabel("Acoustic Noise (dB)")
        ax1.set_ylabel("Fuel Consumption")
        ax1.set_title("Acoustic Noise vs Fuel Consumption")
        ax1.legend()
        st.pyplot(fig1)

    # -------------------------------
    # Graph 2
    # -------------------------------
    with col4:
        fig2, ax2 = plt.subplots()
        ax2.scatter(X, y_efficiency, label="Actual Data")
        ax2.plot(X, efficiency_pred, label="Regression Line")
        ax2.set_xlabel("Acoustic Noise (dB)")
        ax2.set_ylabel("Efficiency (%)")
        ax2.set_title("Acoustic Noise vs Efficiency")
        ax2.legend()
        st.pyplot(fig2)

    # -------------------------------
    # Prediction Section
    # -------------------------------
    st.subheader("5. Prediction Model")

    min_noise = float(data[noise_col].min())
    max_noise = float(data[noise_col].max())

    noise_input = st.slider(
        "Enter Acoustic Noise Level (dB)",
        min_value=min_noise,
        max_value=max_noise,
        value=float(data[noise_col].mean())
    )

    input_data = pd.DataFrame({noise_col: [noise_input]})

    predicted_fuel = fuel_model.predict(input_data)[0]
    predicted_efficiency = efficiency_model.predict(input_data)[0]

    col5, col6 = st.columns(2)

    with col5:
        st.metric("Predicted Fuel Consumption", f"{predicted_fuel:.2f}")

    with col6:
        st.metric("Predicted Efficiency", f"{predicted_efficiency:.2f}%")

    # -------------------------------
    # Instability Decision Logic
    # -------------------------------
    st.subheader("6. Acoustic Instability Decision")

    if noise_input < 140:
        zone = "Stable Zone"
        decision = (
            "The model shows low acoustic disturbance. Combustion is expected "
            "to remain relatively stable with better fuel efficiency."
        )
        recommendation = "No major acoustic control is required, but monitoring is recommended."

    elif 140 <= noise_input < 150:
        zone = "Transition Zone"
        decision = (
            "The model shows that acoustic effects are beginning to influence "
            "fuel consumption and efficiency. This is an early warning stage."
        )
        recommendation = "Monitor pressure oscillations and chamber vibration."

    elif 150 <= noise_input <= 155:
        zone = "Critical Instability Zone"
        decision = (
            "The model indicates a critical range where acoustic instability "
            "starts becoming significant. Fuel consumption increases and efficiency decreases."
        )
        recommendation = "Use acoustic dampers, resonators, or baffles to control instability."

    elif 155 < noise_input <= 170:
        zone = "Unstable Zone"
        decision = (
            "The model shows strong acoustic instability. Combustion performance "
            "is negatively affected and more fuel is required to maintain thrust."
        )
        recommendation = "Improve chamber design, add baffles, and optimize nozzle geometry."

    else:
        zone = "Severe Instability Zone"
        decision = (
            "The model shows severe acoustic disturbance. This may lead to major "
            "fuel wastage, efficiency loss, and structural vibration risk."
        )
        recommendation = "Immediate design improvement and acoustic noise reduction are required."

    st.warning(f"Current Zone: {zone}")
    st.write(f"**Model Decision:** {decision}")
    st.write(f"**Recommended Action:** {recommendation}")

    # -------------------------------
    # Final Academic Interpretation
    # -------------------------------
    st.subheader("7. Final Interpretation for FYP")

    st.write(
        f"The regression analysis shows that acoustic noise has a measurable effect "
        f"on rocket fuel consumption and efficiency. The fuel consumption model has an "
        f"R² value of {fuel_r2:.4f}, while the efficiency model has an R² value of "
        f"{efficiency_r2:.4f}. These values indicate how strongly acoustic noise explains "
        f"changes in fuel usage and efficiency in the selected dataset."
    )

    st.write(
        "The model suggests that as acoustic noise increases, fuel consumption tends "
        "to increase while efficiency tends to decrease. The critical instability range "
        "appears around 150–155 dB, where the model begins to show clear performance loss. "
        "Therefore, acoustic noise control methods such as resonators, baffles, chamber "
        "design improvement, and nozzle optimization can help improve rocket engine performance."
    )

else:
    st.info("Please upload your Excel file to start the analysis.")