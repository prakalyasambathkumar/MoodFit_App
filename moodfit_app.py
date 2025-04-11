import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import requests

# Load dataset
df = pd.read_csv(r"C:\Users\prakalya\OneDrive\Desktop\project_new\moodfit_dataset.csv")

# Encode categorical columns
le_mood = LabelEncoder()
le_weather = LabelEncoder()
le_workout = LabelEncoder()

df["Mood"] = le_mood.fit_transform(df["Mood"])
df["Weather"] = le_weather.fit_transform(df["Weather"])
df["Workout"] = le_workout.fit_transform(df["Workout"])

X = df[["Mood", "Weather", "Temperature", "Humidity", "WindSpeed"]]
y = df["Workout"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Reverse map for predictions
reverse_workout_map = dict(zip(df["Workout"], le_workout.inverse_transform(df["Workout"])))

# Streamlit app
st.title("🏋️ MoodFit: Workout Recommender")

# Mood input
user_mood = st.selectbox("How do you feel right now?", le_mood.classes_)

# Location input for weather
location = st.text_input("Enter your city (for real-time weather)", value="Chennai")

# Get weather using OpenWeatherMap API
API_KEY = "46c3db1818a152675f1e3bf5002d7315"  # Replace with your real API key from openweathermap.org
weather_data = {}

if location:
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={API_KEY}&units=metric"
        res = requests.get(url).json()
        weather_data = {
            "Temperature": res["main"]["temp"],
            "Humidity": res["main"]["humidity"],
            "WindSpeed": res["wind"]["speed"],
            "Weather": res["weather"][0]["main"]
        }
        st.success(f"Weather in {location.title()}: {weather_data['Weather']}, {weather_data['Temperature']}°C")
    except:
        st.warning("Failed to fetch weather. Using default values.")
        weather_data = {"Temperature": 25, "Humidity": 50, "WindSpeed": 5, "Weather": "Sunny"}

# Predict button
if st.button("Suggest Workout"):
    mood_encoded = le_mood.transform([user_mood])[0]
    weather_clean = weather_data["Weather"]
    
    # Map unknown weather types to known types
    known_weather = le_weather.classes_
    weather_final = weather_clean if weather_clean in known_weather else "Sunny"
    weather_encoded = le_weather.transform([weather_final])[0]

    features = [[mood_encoded, weather_encoded,
                 weather_data["Temperature"],
                 weather_data["Humidity"],
                 weather_data["WindSpeed"]]]

    prediction = model.predict(features)[0]
    workout = le_workout.inverse_transform([prediction])[0]

    st.subheader(f"💡 Recommended Workout: **{workout}**")
