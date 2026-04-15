from unittest import result

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier  # is good for predicting categories like Low/Moderate/High risk.
from sklearn.metrics import (
    mean_squared_error,    #measure average error for regression.
    r2_score,       #measure how well model explains the data.
    accuracy_score,  #measure what % of classification were correct.
    classification_report, # gives detailed breakdown per category.
)
from sklearn.preprocessing import LabelEncoder
def assign_risk_level(aqi_value):
    if aqi_value <= 50:
        return "Low"
    elif aqi_value <= 100:
        return "Moderate"
    elif aqi_value <= 200:
        return "High"
    else:
        return "Very High"
    
def train_aqi_regression_model(X,y):
    print("\nTrianing the AQI regression model...")
    print(f"Working with{X.shape[0]} data points")

    X_train, X_test, y_train, y_test = train_test_split(
        X,y, test_size = 0.2, random_state = 42
    )    
    print(f"Training on{len(X_train)} samples, testing on{len(X_test)} samples")
    model = LinearRegression()
    model.fit(X_train , y_train)
    print("Regression model finished training!")
    predictions = model.predict (X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print("\n--- Regression Model Results ---")
    print(f"MSE : {mse:.2f} (Lower is better)")
    print(f"RMSE: {rmse:.2f} (My predictions are off by ~ {rmse:.0f} AQI units on average)")
    print(f"R2 : {r2:.2f} (closer to 1.0 means better fit)")
    return model
def train_health_risk_classifier(X,y):
    print("\nTraining the health risk classifier...")
    print(f"Working with {X.shape[0]} data points")

    X_train, X_test, y_train, y_test = train_test_split(
        X,y, test_size = 0.2, random_state = 42
    )
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
    model = RandomForestClassifier(n_estimators = 100, random_state = 42)
    model.fit(X_train, y_train)
    print("Classifier finished Training!")
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)   #accuracy is number of correct predictions didvided by total predictions.
    print(f"\nAccuracy : {accuracy:.2f} which means  {accuracy*100:.1f}% of predictions were correct")
    print("\nDetailed result for each risk category:")
    print(classification_report(y_test, predictions))
    return model

def predict_health_risk(model,encoder,temperature,humidity):
    input_values = np.array([[temperature, humidity]])
    predicted_number = model.predict(input_values)[0]

    risk_category = encoder.inverse_transform([predicted_number])[0]

    advice = {
        "Low" : "Air quality is good! Safe to go outside and exercise freely.",
        "Moderate" : "Air quality is okay, but if you have asthama or allergies be little careful.",
        "High" : "Air quality is not great today. Try to wear a mask if going outside and avoid heavy exercise.",
        "Very High" : "Air quality is really bad . Best to stay indoors, use an air purifier if you have one"
    }
    return risk_category,advice.get (risk_category,"Stay aware of your Surroundings and health.")
def get_health_advice(risk_level, temperature, humidity, aqi):
    pass


#Quick function test I ran while building this
# I tested assign_risk_level() manually to make sure
# it returns the correct category for different AQI values

#result 1
#result  = assign_risk_level(85)
#print("Risk Level:",result)
#Output I got : "Moderate"
#result 2
#result  = assign_risk_level(120)
#print("Risk Level:",result)
#Output I got : "High"

#result 3
#result  = assign_risk_level(45)
#print("Risk Level:",result)
#Output I got : "Low"

#result 4
#result = assign_risk_level(300)
#print("Risk Level:",result)
#Output I got : "Very High"