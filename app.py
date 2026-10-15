import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Can I Play Football?", page_icon="⚽", layout="centered")

FEATURES = ["Temperature", "Humidity", "Rain"]


@st.cache_resource
def train_model():
    # Same steps as the notebook: load, encode, create target, split, scale, KNN (K=3)
    df = pd.read_csv("weather_forecast_data.csv")[FEATURES].dropna().drop_duplicates()
    df["Rain"] = df["Rain"].map({"rain": 1, "no rain": 0})
    df["Play_Outside"] = (
        (df["Rain"] == 0)
        & (df["Temperature"].between(15, 32))
        & (df["Humidity"] <= 85)
    ).astype(int)

    X, y = df[FEATURES], df["Play_Outside"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_scaled, y_train)
    accuracy = accuracy_score(y_test, knn.predict(scaler.transform(X_test)))
    return df, X_train, y_train, scaler, knn, accuracy


df, X_train, y_train, scaler, knn, accuracy = train_model()

st.title("⚽ Can I Play Football Outside?")
st.write(
    "Enter today's weather and a **K-Nearest Neighbors (K = 3)** model will decide "
    "whether it's a good day to play football."
)

# ---- Inputs ----
st.sidebar.header("🌤️ Today's Weather")
temperature = st.sidebar.slider("Temperature (°C)", 10.0, 35.0, 24.0, 0.5)
humidity = st.sidebar.slider("Humidity (%)", 30.0, 100.0, 55.0, 1.0)
rain = st.sidebar.radio("Is it raining?", ["No", "Yes"], horizontal=True)
rain_value = 1 if rain == "Yes" else 0

# ---- Prediction ----
new_day = pd.DataFrame([[temperature, humidity, rain_value]], columns=FEATURES)
new_day_scaled = scaler.transform(new_day)
prediction = knn.predict(new_day_scaled)[0]
distances, indices = knn.kneighbors(new_day_scaled)

if prediction == 1:
    st.success("## ✅ YES — go play football!")
else:
    st.error("## ❌ NO — better stay inside.")

c1, c2, c3 = st.columns(3)
c1.metric("Temperature", f"{temperature} °C")
c2.metric("Humidity", f"{humidity:.0f} %")
c3.metric("Rain", rain)

# ---- Neighbors (explain the vote) ----
st.subheader("🔍 How the model decided")
st.write("It found the **3 most similar** past weather observations and took a majority vote:")
neighbors = X_train.iloc[indices[0]].copy()
neighbors["Rain"] = neighbors["Rain"].map({1: "Yes", 0: "No"})
neighbors["Play_Outside"] = y_train.iloc[indices[0]].map({1: "✅ Yes", 0: "❌ No"}).values
neighbors["Distance"] = distances[0].round(3)
st.dataframe(neighbors.reset_index(drop=True), width="stretch")
yes_votes = int(y_train.iloc[indices[0]].sum())
st.write(f"**Votes →** Yes: {yes_votes} · No: {3 - yes_votes}")

# ---- Chart ----
st.subheader("📊 Where today sits in the data")
fig, ax = plt.subplots(figsize=(7, 4))
colors = df["Play_Outside"].map({1: "#5cb85c", 0: "#d9534f"})
ax.scatter(df["Temperature"], df["Humidity"], c=colors, s=8, alpha=0.5)
ax.scatter(temperature, humidity, c="blue", s=250, marker="*", edgecolors="black", label="Today")
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Humidity (%)")
ax.set_title("Green = Play, Red = Don't play")
ax.legend()
st.pyplot(fig)

# ---- About ----
with st.expander("ℹ️ About this project"):
    st.markdown(
        f"""
- **Dataset:** {len(df):,} weather observations
- **Features:** Temperature, Humidity, Rain
- **Target:** Play_Outside (Yes if no rain, 15–32 °C, humidity ≤ 85 %)
- **Model:** K-Nearest Neighbors, K = 3, with feature scaling
- **Train/Test split:** 80 % / 20 %
- **Test accuracy:** {accuracy * 100:.2f} %
"""
    )
