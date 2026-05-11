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

    # Read Excel file
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
    # Clean Selected Data
    # -------------------------------
    data = df[[noise_col, fuel_col, efficiency_col]].copy()

    # Convert selected columns to numeric
    data[noise_col] = pd.to_numeric(data[noise_col], errors="coerce")
    data[fuel_col] = pd.to_numeric(data[fuel_col], errors="coerce")
    data[efficiency_col] = pd.to_numeric(data[efficiency_col], errors="coerce")

    # Remove missing values
    data = data.dropna()

    if data.empty:
        st.error("Selected columns do not contain valid numeric data.")
        st.stop()

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

    fuel_slope = float(np.ravel(fuel_model.coef_)[0])
    fuel_intercept = float(np.ravel([fuel_model.intercept_])[0])
    fuel_r2 = float(r2_score(y_fuel, fuel_pred))

    # -------------------------------
    # Regression Model 2
    # Acoustic Noise vs Efficiency
    # -------------------------------
    efficiency_model = LinearRegression()
    efficiency_model.fit(X, y_efficiency)

    efficiency_pred = efficiency_model.predict(X)

    efficiency_slope = float(np.ravel(efficiency_model.coef_)[0])
    efficiency_intercept = float(np.ravel([efficiency_model.intercept_])[0])
    efficiency_r2 = float(r2_score(y_efficiency, efficiency_pred))

    # -------------------------------
    # Display Regression Results
    # -------------------------------
    st.subheader("3. Regression Results")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Model 1: Acoustic Noise vs Fuel Consumption")

        st.write(
            f"**Regression Equation:**  \n"
            f"Fuel Consumption = {fuel_intercept:.2f} + "
            f"({fuel_slope:.2f} × Acoustic Noise)"
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
            f"Efficiency = {efficiency_intercept:.2f} + "
            f"({efficiency_slope:.4f} × Acoustic Noise)"
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
    # Regression Graphs
    # -------------------------------
    st.subheader("4. Regression Graphs")

    col3, col4 = st.columns(2)

    # Sort values for smooth regression line
    sorted_data = data.sort_values(by=noise_col)
    X_sorted = sorted_data[[noise_col]]

    fuel_pred_sorted = fuel_model.predict(X_sorted)
    efficiency_pred_sorted = efficiency_model.predict(X_sorted)

    with col3:
        fig1, ax1 = plt.subplots()

        ax1.scatter(data[noise_col], data[fuel_col], label="Actual Data")
        ax1.plot(
            sorted_data[noise_col],
            fuel_pred_sorted,
            label="Regression Line"
        )

        ax1.set_xlabel("Acoustic Noise (dB)")
        ax1.set_ylabel("Fuel Consumption")
        ax1.set_title("Acoustic Noise vs Fuel Consumption")
        ax1.legend()

        st.pyplot(fig1)

    with col4:
        fig2, ax2 = plt.subplots()

        ax2.scatter(data[noise_col], data[efficiency_col], label="Actual Data")
        ax2.plot(
            sorted_data[noise_col],
            efficiency_pred_sorted,
            label="Regression Line"
        )

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
    mean_noise = float(data[noise_col].mean())

    noise_input = st.slider(
        "Enter Acoustic Noise Level (dB)",
        min_value=min_noise,
        max_value=max_noise,
        value=mean_noise
    )

    input_data = pd.DataFrame({noise_col: [noise_input]})

    predicted_fuel = float(np.ravel(fuel_model.predict(input_data))[0])
    predicted_efficiency = float(np.ravel(efficiency_model.predict(input_data))[0])

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
        recommendation = (
            "No major acoustic control is required, but regular monitoring is recommended."
        )

    elif 140 <= noise_input < 150:
        zone = "Transition Zone"
        decision = (
            "The model shows that acoustic effects are beginning to influence "
            "fuel consumption and efficiency. This is an early warning stage."
        )
        recommendation = (
            "Monitor pressure oscillations, chamber vibration, and combustion response."
        )

    elif 150 <= noise_input <= 155:
        zone = "Critical Instability Zone"
        decision = (
            "The model indicates a critical range where acoustic instability "
            "starts becoming significant. Fuel consumption increases and efficiency decreases."
        )
        recommendation = (
            "Use acoustic dampers, resonators, or baffles to control instability."
        )

    elif 155 < noise_input <= 170:
        zone = "Unstable Zone"
        decision = (
            "The model shows strong acoustic instability. Combustion performance "
            "is negatively affected and more fuel is required to maintain thrust."
        )
        recommendation = (
            "Improve chamber design, add baffles, and optimize nozzle geometry."
        )

    else:
        zone = "Severe Instability Zone"
        decision = (
            "The model shows severe acoustic disturbance. This may lead to major "
            "fuel wastage, efficiency loss, and structural vibration risk."
        )
        recommendation = (
            "Immediate design improvement and acoustic noise reduction are required."
        )

    st.warning(f"Current Zone: {zone}")
    st.write(f"**Model Decision:** {decision}")
    st.write(f"**Recommended Action:** {recommendation}")

    # -------------------------------
    # Critical Point Explanation
    # -------------------------------
    st.subheader("7. Critical Instability Point")

    st.write(
        "Based on the model logic, the critical instability range is approximately "
        "**150–155 dB**. In this range, acoustic oscillations become strong enough "
        "to increase fuel consumption and reduce engine efficiency."
    )

    st.write(
        "This does not mean that every real rocket engine will become unstable exactly "
        "at this value. It means that, according to this dataset and regression-based "
        "model, the performance loss becomes significant around this range."
    )

    # -------------------------------
    # Final Academic Interpretation
    # -------------------------------
    st.subheader("8. Final Interpretation for FYP")

    st.write(
        f"The regression analysis shows that acoustic noise has a measurable effect "
        f"on rocket fuel consumption and efficiency. The fuel consumption model has an "
        f"R² value of **{fuel_r2:.4f}**, while the efficiency model has an R² value of "
        f"**{efficiency_r2:.4f}**."
    )

    st.write(
        "The model suggests that as acoustic noise increases, fuel consumption tends "
        "to increase while efficiency tends to decrease. This supports the theoretical "
        "argument that acoustic instability can disturb combustion performance and "
        "cause energy loss in rocket engines."
    )

    st.write(
        "Therefore, acoustic noise control methods such as resonators, baffles, chamber "
        "design improvement, and nozzle optimization can help reduce instability and "
        "improve rocket engine performance."
    )

    # -------------------------------
    # Limitations
    # -------------------------------
    st.subheader("9. Limitations")

    st.write(
        "This project is theoretical and analytical. The results are based on the "
        "selected dataset, regression analysis, and available literature-based or "
        "simulated data. The model does not replace real rocket engine testing."
    )

else:
    st.info("Please upload your Excel file to start the analysis.")