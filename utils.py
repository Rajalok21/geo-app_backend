import pandas as pd
import re, json, os
import osmnx as ox
from shapely.geometry import Point
from huggingface_hub import InferenceClient

HF_TOKEN = os.getenv("HF_TOKEN")  # Use Render env variable
client = InferenceClient("mistralai/Mistral-7B-Instruct-v0.2", token=HF_TOKEN)

# Load data once
housing_df = pd.read_csv("housing_data.csv")
housing_df.columns = [col.strip() for col in housing_df.columns]
housing_df.rename(columns={
    "Area_Name": "area_name",
    "Pucca": "pucca",
    "Semi_Pucca": "semi_Pucca",
    "Kutcha": "kutcha",
    "Latitude": "latitude",
    "Longitude": "longitude",
    "LULC_Code": "LULC_Code",
    "LULC_Description": "LULC_Description"
}, inplace=True)
housing_df["LULC_Description"] = housing_df["LULC_Description"].str.strip()
housing_df = housing_df[housing_df['area_name'].str.contains("Bangalore", case=False, na=False)]

def llm_extract(question):
    prompt = f"""
You are an assistant that extracts structured search parameters from location-based questions.

Extract and return:
- "amenity" (optional)
- "location"
- "radius" in meters (default 2000)
- "housing_type" (e.g., kutcha, pucca, semi_Pucca) [optional]
- "lulc" (e.g., Cropland, Built-up) [optional]

Respond in JSON format only:
{{
  "amenity": "hospital",
  "location": "MG Road",
  "radius": 2000,
  "housing_type": "kutcha",
  "lulc": "Cropland"
}}

Query: {question}
"""

    response = client.chat_completion(messages=[
        {"role": "system", "content": "You are a helpful assistant that extracts structured location-based search queries."},
        {"role": "user", "content": prompt}
    ])
    output = response.choices[0].message["content"].strip()

    json_match = re.search(r'{[\s\S]*?}', output)
    if not json_match:
        return None, None, 2000, None, None

    parsed = json.loads(json_match.group())
    amenity = str(parsed.get("amenity", "")).strip()
    location = str(parsed.get("location", "")).strip()
    radius = int(parsed.get("radius", 2000))
    housing_type_raw = str(parsed.get("housing_type", "")).lower().strip()
    housing_type_map = {
        "pucca": "pucca",
        "kutcha": "kutcha",
        "semi pucca": "semi_Pucca",
        "semi_pucca": "semi_Pucca"
    }
    housing_type = housing_type_map.get(housing_type_raw, "")
    lulc = str(parsed.get("lulc", "")).strip()
    return amenity, location, radius, housing_type, lulc

def get_data_from_query(query):
    amenity, location, radius, housing_type, lulc = llm_extract(query)

    # Geocode location
    coords = re.findall(r"[-+]?\d*\.\d+|\d+", location or query)
    if len(coords) >= 2:
        lat, lon = float(coords[0]), float(coords[1])
    else:
        lat, lon = ox.geocode(location + ", Bangalore")

    result = {
        "query": query,
        "location": location,
        "lat": lat,
        "lon": lon,
        "radius": radius,
        "amenity": amenity,
        "housing_type": housing_type,
        "lulc": lulc,
        "housing_matches": []
    }

    filtered_df = housing_df.copy()
    if housing_type and housing_type in filtered_df.columns:
        filtered_df = filtered_df[filtered_df[housing_type] > 0]
    if lulc:
        filtered_df = filtered_df[
            filtered_df["LULC_Description"].str.lower().str.contains(lulc.lower(), na=False)
        ]

    for _, row in filtered_df.iterrows():
        result["housing_matches"].append({
            "area": row["area_name"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "percentage": row.get(housing_type, None),
            "lulc": row["LULC_Description"]
        })

    return result
