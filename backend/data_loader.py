import pandas as pd

# os helps me build file paths that work on any computer.
import os

from sklearn.preprocessing import LabelEncoder
def load_data(file_path):
    print("loading my climate health dataset...")
    data= pd.read_csv(file_path)
    print(f"Dataset loaded! I have {data.shape[0]} rows and {data.shape[1]} columns.")
    data.columns = data.columns.str.strip()
    for col in data.columns:
        if data[col].dtype == "object":
            data[col] = data[col].str.strip()


            city_corrections = {
                "Banglore" : "Bangalore",
                "Hydrabad" : "Hyderabad",
                "Bhubneswar" : "Bhubaneswar"
            }
    data["City"] = data["City"].replace(city_corrections)
    print("fixed city name typos in the dataset.")
    # errors="coerce" means if something can't be converted, make it NaN instead of crashing(NAN means "Not a Number").
    data["Temperature(°C)"] = pd.to_numeric(data["Temperature(°C)"],errors="coerce")
    data["Humidity(%)"] = pd.to_numeric(data["Humidity(%)"], errors= "coerce")
    data["Air Quality Index(AQI) Value"] = pd.to_numeric(data["Air Quality Index(AQI) Value"], errors="coerce")
    before = data.shape[0]
    data = data.dropna(subset = ["Temperature(°C)", "Humidity(%)", "Air Quality Index(AQI) Value"])
    after = data.shape[0]
    print(f"Removed {before - after} rowa with missing values.{after} rows remaining.")
    return data

def get_features (data):
    features = data [["Temperature(°C)","Humidity(%)"]]
    print(f"Selected{features.shape[1]} input features: Temperature and Humidity")
    return features



def get_aqi_values(data):
    aqi = data["Air Quality Index(AQI) Value"]

    print(f"AQI value ready-range is {aqi.min():.0f} to {aqi.max():.0f}")
    return aqi

def encode_aqi_category(data):
    encoder = LabelEncoder()
    data["AQI_Category_Encoded"] = encoder.fit_transform(data["AQI Category"])
    # printing the mapping so I can understand what number means what.
    print("AQI categories converted to numbers:")
    for number, category in enumerate(encoder.classes_):
        print(f"{number} = {category}")
        return data["AQI_Category_Encoded"],encoder
#chart 1 - average AQI per city for the bar chart.
def get_dashboard_data(data):
    aqi_by_city = (data.groupby("City")["Air Quality Index(AQI) Value"]
                   .mean() #calculates the average AQI for each city group.
                   .round(1)  # rounding to 1 decimal place looks cleaner
                   .reset_index()   # this turns the city names back into a regular column.
               .sort_values("Air Quality Index(AQI) Value",ascending = False) # highest AQI first.
    )

    #chart 2 - Temperature vs AQI scatter plot data.
    scatter = data[["Temperature(°C)", "Air Quality Index(AQI) Value","City"]].copy()
    scatter = scatter.dropna()
    scatter = scatter.rename(columns = {
        "Temperature(°C)" :"temp",
        "Air Quality Index(AQI) Value" : "aqi",
    })

    #chart 3 - How many lines falls in each risk category for the pie chart.
    # firstly  I need to assign a risk level based on AQI values.

    def assign_risk(aqi_val):
        if aqi_val <=50 : return "Low"
        elif aqi_val <=100 : return "Moderate"
        elif aqi_val <=200 : return "High"
        else               : return "Very High"

    data["Risk Level"] = data["Air Quality Index(AQI) Value"].apply(assign_risk)
    risk_counts = data["Risk Level"].value_counts().reset_index()
    risk_counts.columns = ["category", "count"]
    return {
        "aqi_by_city": aqi_by_city.to_dict(orient = "records"),  #.to_dict(orient = "records") converts the DataFrame into a list of dictionaries.
        "scatter"    : scatter.to_dict(orient = "records"),
        "risk_distribution" : risk_counts.to_dict(orient = "records")
    }

