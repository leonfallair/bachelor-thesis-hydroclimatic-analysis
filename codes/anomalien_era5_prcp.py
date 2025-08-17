import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from rasterio.features import geometry_mask
from scipy.stats import linregress
import pandas as pd

# 1. Daten laden
file_path = "data/era5_prcp_1950_2025.grib"
vector_path = "qgis_data/gadm41_UZB.gpkg"

# Daten für avgad
ds = xr.open_dataset(file_path, engine="cfgrib", filter_by_keys={'stepType': 'avgad'})
# Daten für avgas
ds2 = xr.open_dataset(file_path, engine="cfgrib", filter_by_keys={'stepType': 'avgas'})

# Kombinieren der Zeitreihen
combined_precip = xr.concat([ds, ds2], dim="time")
combined_precip = combined_precip.sortby('time')

# Relevante Daten auswählen: Niederschlag (total precipitation) in mm
precipitation = combined_precip['tp'] * 25000  # Umrechnung von Metern auf Millimeter

# Eingrenzen der Zeitreihe (1950-2023)
precipitation = precipitation.sel(time=slice("1950-01-01", "2024-12-01"))

# Monatliche Daten in jährliche Summen umwandeln
precipitation_annual = precipitation.resample(time="Y").sum(dim="time")

# 2. Vektordaten laden und Maske erstellen
gdf = gpd.read_file(vector_path, layer="ADM_ADM_0")
usbekistan_geometry = gdf.geometry.union_all()  # Geometrien vereinigen

# Rasterkoordinaten und Transform erstellen
lat, lon = precipitation_annual.latitude.values, precipitation_annual.longitude.values
transform = precipitation_annual.rio.transform()

# Maske basierend auf dem Vektorlayer
mask = geometry_mask(
    [usbekistan_geometry],
    transform=transform,
    invert=True,
    out_shape=(len(lat), len(lon))
)

# Maske anwenden
mask_da = xr.DataArray(mask, coords={"latitude": lat, "longitude": lon}, dims=["latitude", "longitude"])
precipitation_annual_masked = precipitation_annual.where(mask_da)

# 3. Durchschnittlicher Niederschlag für Usbekistan (jährlich)
precip_uzbekistan_mean = precipitation_annual_masked.mean(dim=["latitude", "longitude"])

# 4. Berechnung der Niederschlaganomalien
reference_period = (1961, 1990)
mask_reference = (precip_uzbekistan_mean['time'].dt.year >= reference_period[0]) & (precip_uzbekistan_mean['time'].dt.year <= reference_period[1])
reference_mean = precip_uzbekistan_mean.sel(time=mask_reference).mean().values

precip_anomalies = precip_uzbekistan_mean - reference_mean

# 5. Visualisierung der Niederschlaganomalien
plt.figure(figsize=(12, 7))

# Bestimmung der Farben für die Balken (rot für positive Anomalien, blau für negative)
colors = ['red' if anomaly < 0 else 'blue' for anomaly in precip_anomalies.values.flatten()]

# Balkendiagramm zeichnen
plt.bar(precip_anomalies['time'].dt.year, precip_anomalies, color=colors)

plt.axhline(0, color='black', linewidth=1)
plt.xlabel('Jahr')
plt.ylabel('Niederschlag Anomalie (mm)')
plt.title('Niederschlagsanomalien in Usbekistan (1950–2024) mit Referenzperiode (1961–1990)', fontsize=16, fontweight='bold')
plt.grid(True)

# Legende hinzufügen
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color='red', lw=4, label='Negative Anomalie'),
                   Line2D([0], [0], color='blue', lw=4, label='Positive Anomalie')]
plt.legend(handles=legend_elements, loc='upper left')

# Diagramm anzeigen
plt.savefig("images/anomalien_era5_prcp.png", dpi=300)
plt.show()
