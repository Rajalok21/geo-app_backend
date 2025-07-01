import osmnx as ox
import geopandas as gpd
import pandas as pd
import re
import json
from shapely.geometry import Point
from huggingface_hub import InferenceClient

# ========= Initial Setup ========= #
center_coords = (12.9763, 77.6033)  # MG Road, Bangalore

# Load housing data
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

# Tags for OSM extraction
tags = {
    "amenity": True, "shop": True, "building": True, "railway": True,
    "highway": True, "bus": True, "aeroway": True, "landuse": True,
    "man_made": True, "industrial": True, "power": True,
    "generator:source": True, "waterway": True, "leisure": True
}

# HuggingFace Inference client
client = InferenceClient("mistralai/Mistral-7B-Instruct-v0.2", token="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")

# ========= Function to extract structured query ========= #
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
    try:
        response = client.chat_completion(messages=[
            {"role": "system", "content": "You are a helpful assistant that extracts structured location-based search queries."},
            {"role": "user", "content": prompt}
        ])
        output = response.choices[0].message["content"].strip()
    except Exception as e:
        print(f"❌ LLM API call failed: {e}")
        return None, None, 2000, None, None

    json_match = re.search(r'{[\s\S]*?}', output)
    if not json_match:
        print("❌ No valid JSON found.")
        return None, None, 2000, None, None

    try:
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
    except Exception as e:
        print(f"❌ JSON parsing error: {e}")
        return None, None, 2000, None, None

# ========= Main processing function ========= #
def process_query(query):
    global center_coords, housing_df, tags, client

    amenity, location, radius, housing_type, lulc = llm_extract(query)

    # 📍 Geocode
    try:
        coords = re.findall(r"[-+]?\d*\.\d+|\d+", location or query)
        if len(coords) >= 2:
            lat, lon = float(coords[0]), float(coords[1])
            center = (lat, lon)
        elif location:
            latlon = ox.geocode(location + ", Bangalore")
            center = (latlon[0], latlon[1])
        else:
            center = center_coords
    except:
        center = center_coords

    # 🛣️ Get road edges
    G = ox.graph_from_point(center, dist=radius, network_type='drive')
    edges = ox.graph_to_gdfs(G, nodes=False)
    expected_cols = ["name", "highway", "width", "lanes", "lane:width", "geometry"]
    available_cols = [col for col in expected_cols if col in edges.columns]
    edges = edges[available_cols].copy()

    def estimate_width(row):
        try:
            if pd.notna(row.get("width")):
                return float(row.get("width"))
            elif pd.notna(row.get("lanes")):
                num_lanes = int(str(row.get("lanes")).split(";")[0].strip())
                lane_width = float(str(row.get("lane:width", 3.5)).split(";")[0].strip()) if "lane:width" in row else 3.5
                return round(num_lanes * lane_width, 2)
        except:
            return None

    edges["estimated_width_m"] = edges.apply(estimate_width, axis=1)

    gdf = ox.features_from_point(center, tags=tags, dist=radius)
    gdf = gdf[[col for col in gdf.columns if isinstance(col, str)]]
    tag_cols = [col for col in tags if col in gdf.columns]
    gdf["category"] = gdf[tag_cols].fillna("").agg("".join, axis=1)

    subset = gdf[gdf["category"] == amenity]
    named_amenities = [
        {
            "name": row["name"],
            "latitude": row.geometry.centroid.y if row.geometry.geom_type != "Point" else row.geometry.y,
            "longitude": row.geometry.centroid.x if row.geometry.geom_type != "Point" else row.geometry.x
        }
        for _, row in subset.iterrows() if pd.notna(row.get("name"))
    ]

    # 🏠 Housing filter
    housing_subset = housing_df.copy()
    if housing_type and housing_type in housing_df.columns:
        housing_subset = housing_subset[housing_subset[housing_type] > 0]
    if lulc:
        housing_subset = housing_subset[
            housing_subset["LULC_Description"].str.lower().str.contains(lulc.lower(), na=False)
        ]

    housing_data = []
    for _, row in housing_subset.iterrows():
        housing_data.append({
            "area": row["area_name"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "lulc": row["LULC_Description"],
            "percentage": row.get(housing_type, None) if housing_type else None
        })

    road_data = [
        {
            "name": row.get("name", "Unnamed"),
            "estimated_width_m": row["estimated_width_m"]
        }
        for _, row in edges.iterrows()
        if pd.notna(row.get("estimated_width_m"))
    ]

    return {
        "query": query,
        "amenity": amenity,
        "location": location,
        "lat": center[0],
        "lon": center[1],
        "radius": radius,
        "housing_type": housing_type,
        "lulc": lulc,
        "amenity_matches": named_amenities,
        "housing_matches": housing_data,
        "road_widths": road_data
    }
__all__ = ['process_query']