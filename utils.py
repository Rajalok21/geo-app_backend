import os
import google.generativeai as genai
import osmnx as ox
import geopandas as gpd
import pandas as pd
import re
import json
from shapely.geometry import Point
from dotenv import load_dotenv

# ======== Setup ========
load_dotenv()
genai.configure(api_key=os.getenv("key"))

model = genai.GenerativeModel("gemini-2.0-flash")

center_coords = (12.9763, 77.6033)  # MG Road, Bangalore

# 📦 Load housing data
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

# 🔖 OSM Tags
tags = {
    "amenity": True, "shop": True, "building": True, "railway": True,
    "highway": True, "bus": True, "aeroway": True, "landuse": True,
    "man_made": True, "industrial": True, "power": True,
    "generator:source": True, "waterway": True, "leisure": True
}

# ========= LLM Query Parser =========
def llm_extract(question):
    prompt = f"""
You are an assistant that extracts structured search parameters from location-based questions.

Extract and return:
- "amenity" (optional)
- "location"
- "radius" in meters (default 2000)
- "housing_type" (e.g., kutcha, pucca, semi_Pucca) [optional]
- "lulc" (e.g., Cropland, Built-up) [optional]
- "top_k" (number of top results to return, default 5)

Respond in JSON format only:
{{
  "amenity": "hospital",
  "location": "MG Road",
  "radius": 2000,
  "housing_type": "kutcha",
  "lulc": "Cropland",
  "top_k": 5
}}

Query: {question}
"""

    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
    except Exception as e:
        print("❌ Gemini API Error:", e)
        return None, None, 2000, None, None, 5

    json_match = re.search(r'{[\s\S]*?}', text)
    if not json_match:
        print("❌ No valid JSON found.")
        return None, None, 2000, None, None, 5

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
        top_k = int(parsed.get("top_k", 5))
        return amenity, location, radius, housing_type, lulc, top_k
    except Exception as e:
        print("❌ JSON Parsing Error:", e)
        return None, None, 2000, None, None, 5

# ========= Main Processing =========
def process_query(query):
    global center_coords, housing_df, tags

    amenity, location, radius, housing_type, lulc, top_k = llm_extract(query)

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

    # 🛣️ Get roads
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

    # 🏢 Get OSM Features
    gdf = ox.features_from_point(center, tags=tags, dist=radius)
    gdf = gdf[[col for col in gdf.columns if isinstance(col, str)]]
    tag_cols = [col for col in tags if col in gdf.columns]
    gdf["category"] = gdf[tag_cols].fillna("").agg("".join, axis=1)

    subset = gdf[gdf["category"] == amenity]
    named_amenities = []
    for _, row in subset.iterrows():
        if pd.notna(row.get("name")):
            lat = row.geometry.centroid.y if row.geometry.geom_type != "Point" else row.geometry.y
            lon = row.geometry.centroid.x if row.geometry.geom_type != "Point" else row.geometry.x
            street_view_url = f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}"
            named_amenities.append({
                "name": row["name"],
                "latitude": lat,
                "longitude": lon,
                "street_view_url": street_view_url
            })
            if len(named_amenities) >= top_k:
                break

    # 🏘️ Filter housing data
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

    # 🌐 Query center street view URL
    street_view_url = f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={center[0]},{center[1]}"

    return {
        "query": query,
        "amenity": amenity,
        "location": location,
        "lat": center[0],
        "lon": center[1],
        "radius": radius,
        "housing_type": housing_type,
        "lulc": lulc,
        "top_k": top_k,
        "amenity_matches": named_amenities,
        "housing_matches": housing_data,
        "road_widths": road_data,
        "street_view_url": street_view_url
    }

__all__ = ['process_query']
