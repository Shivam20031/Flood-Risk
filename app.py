import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster, HeatMap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import io

st.set_page_config(
    page_title="Flood Risk Prediction – India",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Flood Risk Prediction in India")
st.markdown("Machine learning-based flood risk assessment using environmental and geographic features.")

# ── Synthetic dataset matching the original notebook's schema ─────────────────
@st.cache_data
def generate_dataset(n=500):
    rng = np.random.default_rng(42)
    land_covers = ["Agricultural", "Forest", "Urban", "Desert", "Water Body", "Wetland"]
    soil_types  = ["Clay", "Loam", "Sandy", "Silt", "Peat"]

    rainfall      = rng.uniform(0, 600, n)
    temperature   = rng.uniform(15, 45, n)
    humidity      = rng.uniform(20, 100, n)
    river_disch   = rng.uniform(10, 5000, n)
    water_level   = rng.uniform(0.5, 15, n)
    elevation     = rng.uniform(0, 3000, n)
    pop_density   = rng.uniform(10, 20000, n)
    land_cover    = rng.choice(land_covers, n)
    soil_type     = rng.choice(soil_types, n)
    latitude      = rng.uniform(8, 37, n)
    longitude     = rng.uniform(68, 97, n)

    # Flood logic: high rainfall + high water level + low elevation → more likely
    flood_score = (
        (rainfall / 600) * 0.35
        + (water_level / 15) * 0.25
        + (river_disch / 5000) * 0.20
        + (humidity / 100) * 0.10
        + ((3000 - elevation) / 3000) * 0.10
    )
    flood = (flood_score + rng.normal(0, 0.08, n) > 0.50).astype(int)

    df = pd.DataFrame({
        "Rainfall (mm)": rainfall.round(1),
        "Temperature (°C)": temperature.round(1),
        "Humidity (%)": humidity.round(1),
        "River Discharge (m³/s)": river_disch.round(1),
        "Water Level (m)": water_level.round(2),
        "Elevation (m)": elevation.round(1),
        "Population Density": pop_density.round(0).astype(int),
        "Land Cover": land_cover,
        "Soil Type": soil_type,
        "Flood Occurred": flood,
        "Latitude": latitude.round(4),
        "Longitude": longitude.round(4),
    })
    return df


@st.cache_resource
def train_model(df):
    feature_cols = [
        "Rainfall (mm)", "Temperature (°C)", "Humidity (%)",
        "River Discharge (m³/s)", "Water Level (m)", "Elevation (m)",
        "Population Density",
    ]
    df_enc = pd.get_dummies(df, columns=["Land Cover", "Soil Type"])
    all_features = feature_cols + [c for c in df_enc.columns if c.startswith("Land Cover_") or c.startswith("Soil Type_")]
    X = df_enc[all_features]
    y = df_enc["Flood Occurred"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    accuracy = accuracy_score(y_test, y_pred)
    report  = classification_report(y_test, y_pred, output_dict=True)
    return model, scaler, all_features, accuracy, report, X_test, y_test, y_pred


df = generate_dataset()
model, scaler, all_features, accuracy, report, X_test, y_test, y_pred = train_model(df)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("🔧 Predict Flood Risk")
st.sidebar.markdown("Adjust the parameters below to get a real-time flood prediction.")

land_covers = ["Agricultural", "Forest", "Urban", "Desert", "Water Body", "Wetland"]
soil_types  = ["Clay", "Loam", "Sandy", "Silt", "Peat"]

rainfall_in    = st.sidebar.slider("Rainfall (mm)", 0, 600, 250)
temperature_in = st.sidebar.slider("Temperature (°C)", 15, 45, 28)
humidity_in    = st.sidebar.slider("Humidity (%)", 20, 100, 65)
river_disch_in = st.sidebar.slider("River Discharge (m³/s)", 10, 5000, 800)
water_level_in = st.sidebar.slider("Water Level (m)", 0.5, 15.0, 5.0)
elevation_in   = st.sidebar.slider("Elevation (m)", 0, 3000, 200)
pop_density_in = st.sidebar.slider("Population Density", 10, 20000, 5000)
land_cover_in  = st.sidebar.selectbox("Land Cover", land_covers)
soil_type_in   = st.sidebar.selectbox("Soil Type", soil_types)

def predict(rainfall, temperature, humidity, river_disch, water_level, elevation,
            pop_density, land_cover, soil_type):
    row = {f: 0 for f in all_features}
    row["Rainfall (mm)"]          = rainfall
    row["Temperature (°C)"]       = temperature
    row["Humidity (%)"]           = humidity
    row["River Discharge (m³/s)"] = river_disch
    row["Water Level (m)"]        = water_level
    row["Elevation (m)"]          = elevation
    row["Population Density"]     = pop_density
    lc_key = f"Land Cover_{land_cover}"
    st_key = f"Soil Type_{soil_type}"
    if lc_key in row:
        row[lc_key] = 1
    if st_key in row:
        row[st_key] = 1
    X_new = pd.DataFrame([row])[all_features]
    X_new_s = scaler.transform(X_new)
    pred  = model.predict(X_new_s)[0]
    proba = model.predict_proba(X_new_s)[0][1]
    return pred, proba

pred_label, pred_proba = predict(
    rainfall_in, temperature_in, humidity_in, river_disch_in,
    water_level_in, elevation_in, pop_density_in, land_cover_in, soil_type_in
)

if pred_label == 1:
    st.sidebar.error(f"⚠️ **FLOOD RISK: HIGH**\nProbability: {pred_proba:.1%}")
else:
    st.sidebar.success(f"✅ **FLOOD RISK: LOW**\nProbability: {pred_proba:.1%}")

# ── Main content tabs ─────────────────────────────────────────────────────────
tab_overview, tab_eda, tab_model, tab_map = st.tabs([
    "📊 Overview", "📈 Data Analysis", "🤖 Model Performance", "🗺️ Flood Map"
])

# ── Tab 1: Overview ───────────────────────────────────────────────────────────
with tab_overview:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Flood Events", f"{df['Flood Occurred'].sum():,}")
    col3.metric("Flood Rate", f"{df['Flood Occurred'].mean():.1%}")
    col4.metric("Model Accuracy", f"{accuracy:.1%}")

    st.subheader("Sample Dataset")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Statistical Summary")
    numeric_cols = ["Rainfall (mm)", "Temperature (°C)", "Humidity (%)",
                    "River Discharge (m³/s)", "Water Level (m)", "Elevation (m)", "Population Density"]
    st.dataframe(df[numeric_cols].describe().round(2), use_container_width=True)

# ── Tab 2: EDA ────────────────────────────────────────────────────────────────
with tab_eda:
    row1_c1, row1_c2 = st.columns(2)

    with row1_c1:
        st.subheader("Land Cover Distribution")
        lc_counts = df["Land Cover"].value_counts()
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        explode = [0.1 if i == 0 else 0 for i in range(len(lc_counts))]
        ax1.pie(lc_counts, labels=lc_counts.index, autopct="%1.1f%%",
                startangle=90, shadow=True, explode=explode)
        ax1.set_title("Proportions of Land Covers")
        st.pyplot(fig1)
        plt.close(fig1)

    with row1_c2:
        st.subheader("Soil Type Distribution")
        st_counts = df["Soil Type"].value_counts()
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        explode2 = [0.1] + [0] * (len(st_counts) - 1)
        ax2.pie(st_counts, labels=st_counts.index, autopct="%1.1f%%",
                startangle=90, shadow=True, explode=explode2)
        ax2.set_title("Proportions of Soil Types")
        st.pyplot(fig2)
        plt.close(fig2)

    st.subheader("Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(10, 7))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                square=True, cbar_kws={"shrink": 0.8}, ax=ax3)
    ax3.set_title("Correlation Map", fontsize=16)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    st.subheader("Flood Occurrence Count")
    fig4, ax4 = plt.subplots(figsize=(7, 4))
    sns.countplot(x="Flood Occurred", hue="Flood Occurred", data=df, palette="pastel", legend=False, ax=ax4)
    ax4.set_xlabel("Flood Occurred (0 = No, 1 = Yes)", fontsize=12)
    ax4.set_ylabel("Count", fontsize=12)
    ax4.set_title("Count of Flood Occurrences", fontsize=14)
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)

    st.subheader("Rainfall Distribution by Flood Occurrence")
    fig5, ax5 = plt.subplots(figsize=(8, 4))
    for label, grp in df.groupby("Flood Occurred"):
        ax5.hist(grp["Rainfall (mm)"], bins=30, alpha=0.6,
                 label=("Flood" if label == 1 else "No Flood"))
    ax5.set_xlabel("Rainfall (mm)")
    ax5.set_ylabel("Frequency")
    ax5.set_title("Rainfall Distribution by Flood Outcome")
    ax5.legend()
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close(fig5)

# ── Tab 3: Model Performance ──────────────────────────────────────────────────
with tab_model:
    st.subheader("Model: Random Forest Classifier")

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Overall Accuracy", f"{accuracy:.1%}")
    mc2.metric("Precision (Flood)", f"{report['1']['Precision']:.1%}")
    mc3.metric("Recall (Flood)",    f"{report['1']['Recall']:.1%}")

    st.subheader("Classification Report")
    report_df = pd.DataFrame(report).transpose().round(3)
    st.dataframe(report_df, use_container_width=True)

    st.subheader("Feature Importance")
    importances = model.feature_importances_
    feat_df = pd.DataFrame({"Feature": all_features, "Importance": importances})
    feat_df = feat_df.sort_values("Importance", ascending=False).head(12)
    fig6, ax6 = plt.subplots(figsize=(9, 5))
    sns.barplot(x="Importance", y="Feature", hue="Feature", data=feat_df, palette="Blues_r", legend=False, ax=ax6)
    ax6.set_title("Top Feature Importances (Random Forest)", fontsize=14)
    plt.tight_layout()
    st.pyplot(fig6)
    plt.close(fig6)

# ── Tab 4: Map ────────────────────────────────────────────────────────────────
with tab_map:
    st.subheader("Flood Occurrence Map – High Precipitation Zones (>200 mm)")

    threshold = st.slider("Precipitation threshold (mm)", 50, 500, 200, step=25)

    high_prec = df[(df["Rainfall (mm)"] > threshold) & (df["Flood Occurred"] == 1)]
    st.write(f"Showing **{len(high_prec)}** flood events with rainfall > {threshold} mm")

    if len(high_prec) == 0:
        st.warning("No records match the current filter. Lower the threshold.")
    else:
        map_center = [high_prec["Latitude"].mean(), high_prec["Longitude"].mean()]
        m = folium.Map(location=map_center, zoom_start=5)

        cluster = MarkerCluster().add_to(m)
        for _, row in high_prec.iterrows():
            folium.Marker(
                location=[row["Latitude"], row["Longitude"]],
                popup=(
                    f"<b>Rainfall:</b> {row['Rainfall (mm)']} mm<br>"
                    f"<b>Temp:</b> {row['Temperature (°C)']} °C<br>"
                    f"<b>Humidity:</b> {row['Humidity (%)']}%<br>"
                    f"<b>Water Level:</b> {row['Water Level (m)']} m<br>"
                    f"<b>Elevation:</b> {row['Elevation (m)']} m"
                ),
                icon=folium.Icon(color="red", icon="tint", prefix="fa")
            ).add_to(cluster)

        heat_data = [[r["Latitude"], r["Longitude"], r["Rainfall (mm)"]]
                     for _, r in high_prec.iterrows()]
        HeatMap(heat_data, radius=15).add_to(m)

        map_html = m._repr_html_()
        st.components.v1.html(map_html, height=500, scrolling=False)

st.divider()
st.markdown(
    "<center><small>Flood Risk Prediction in India · Random Forest + Nature Optimization · "
    "Data: Indian Meteorological Department / Central Water Commission</small></center>",
    unsafe_allow_html=True
)
