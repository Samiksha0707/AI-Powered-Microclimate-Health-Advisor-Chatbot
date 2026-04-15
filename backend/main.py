import os
from data_loader import (
    load_data,     #loads and cleans the csv file.
    get_features,   #picks Temperature and Humidity columns.
    get_aqi_values,       #picks the AQI value column.
    encode_aqi_category     #converts AQI category text to numbers.
)
from model import (
    assign_risk_level,
    train_aqi_regression_model,   #trains the regression model.
    train_health_risk_classifier, #trains the classifier model.
    predict_health_risk,   #makes predictions on new data.
    get_health_advice   #generates health advice based on risk level.
)
base_dir = os.path.dirname(__file__)
csv_path = os.path.join(base_dir,"Health_climate_data.csv")

#step1 : load the data
print("=" *55)
print("AI Climate Health Advisor --- Starting Up")
print("=" *55)
data = load_data(csv_path)    #calling load_data()from data_loader.py
print("\nFirst 3 rows of my dataset just to double check:")
print(data.head(3))

#step2 : Preparing inputs and outputs for training
X = get_features(data)
aqi_values = get_aqi_values(data)
encoded_categories,encoder = encode_aqi_category(data)

#step3 : Training Model 1 - AQI Regression
print("\n" + "=" * 55)
print("Training Model 1 : AQI Value Predictor")
print("(Linear Regression)")
print("=" * 55)
regression_model = train_aqi_regression_model(X, aqi_values)

#step4 : Training Model 2 - Health Risk Classifier
print("\n" + "=" * 55)
print("Training Model 2 : Health Risk Classifier")
print("(Random Forest  - 100 decision trees)")
print("=" * 55)
classifier_model = train_health_risk_classifier(X, encoded_categories)

#step5 : Auto generating predictions for every city
print("\n" + "=" * 55)
print("Auto Generated Predictions - All cities")
print("=" * 55)
all_cities = sorted(data["City"].unique())
print(f"\nGenerating predictions for {len(all_cities)}cities...\n")
print(f"{'City':<15} {'Temp(°C)':<10} {'Humidity(%)':<12} {'AQI':<8} {'Risk Level':<12} Advice")
print("-" * 90)
for city in all_cities:
    city_rows = data[data["City"].str.lower() == city.lower()]
    latest = city_rows.iloc[-1]
    temp = latest["Temperature(°C)"]
    humidity = latest["Humidity(%)"]
    aqi = latest["Air Quality Index(AQI) Value"]
    risk_level = predict_health_risk(classifier_model, encoder, temp, humidity)
    advice = get_health_advice(risk_level, temp, humidity, aqi)
    print(f"{city:<15} {temp:<10} {humidity:<12} {aqi:<8} {risk_level:<12} {advice[:40]}...")