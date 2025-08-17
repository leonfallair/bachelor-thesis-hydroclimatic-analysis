import xarray as xr
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from rasterio.features import geometry_mask
import matplotlib.colors as mcolors

# === 1. Daten laden ===
file_path = "data/era5_temp_1950_2024.grib"
vector_path = "qgis_data/gadm41_UZB.gpkg"

data = xr.open_dataset(file_path, engine="cfgrib")
temperature = data['t2m'] - 273.15  # Umwandlung von Kelvin nach Celsius
temperature = temperature.sel(time=slice("1991-01-01", "2020-12-31"))

# === 2. Laden des Vektorlayers für Usbekistan ===
gdf = gpd.read_file(vector_path, layer="ADM_ADM_0")
usbekistan_geometry = gdf.geometry.union_all()

# === 3. Maske erstellen ===
lat, lon = temperature.latitude.values, temperature.longitude.values
lon_2d, lat_2d = np.meshgrid(lon, lat)
transform = temperature.rio.transform()

mask = geometry_mask(
    [usbekistan_geometry],
    transform=transform,
    invert=True,
    out_shape=(len(lat), len(lon))
)
mask_da = xr.DataArray(mask, coords={"latitude": lat, "longitude": lon}, dims=["latitude", "longitude"])
temperature = temperature.where(mask_da)

# === 4. Definition der Jahreszeiten ===
seasons = {
    "Dezember–Februar": [12, 1, 2],
    "März–Mai": [3, 4, 5],
    "Juni–August": [6, 7, 8],
    "September–November": [9, 10, 11]
}

trend_maps = {}

# === 5. Berechnung der Trends für jede Saison ===
years = 2020 - 1991  # Zeitraum in Jahren

for season, months in seasons.items():
    seasonal_temp = temperature.sel(time=temperature.time.dt.month.isin(months))
    seasonal_temp_yearly = seasonal_temp.groupby("time.year").mean(dim="time")
    seasonal_temp_yearly = seasonal_temp_yearly.where(mask_da)

    # Lineare Trendberechnung
    trend_map = seasonal_temp_yearly.polyfit(dim="year", deg=1)["polyfit_coefficients"][0]
    total_change_map = trend_map * 10  # Gesamte Temperaturänderung

    trend_maps[season] = total_change_map

# === 6. Plot erstellen ===
fig, axes = plt.subplots(2, 2, figsize=(10, 10), subplot_kw={"projection": ccrs.Mercator()})
fig.suptitle("Räumliche Temperaturtrends in Usbekistan (1991–2020)", fontsize=16, fontweight="bold")

# Gemeinsame Farbbalken-Skala für Vergleichbarkeit
min_change = min(trend_map.min().values for trend_map in trend_maps.values())
max_change = max(trend_map.max().values for trend_map in trend_maps.values())
levels = np.linspace(min_change, max_change, num=15)

# === 7. Karten zeichnen ===
for ax, (season, total_change_map) in zip(axes.flat, trend_maps.items()):
    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())

    trend_plot = ax.contourf(
        total_change_map.longitude,
        total_change_map.latitude,
        total_change_map,
        levels=[-0.25, -0.125, 0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1, 1.125, 1.25, 1.375, 1.5],
        cmap="coolwarm",
        norm=mcolors.TwoSlopeNorm(vmin=-0.25, vcenter=0, vmax=1.5),  # 0°C als Übergangspunkt
        transform=ccrs.PlateCarree()
    )

    ax.set_title(season, fontsize=12, fontweight="bold")

# === 8. Gemeinsamen Farbbalken hinzufügen ===
cbar = fig.colorbar(trend_plot, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.05, pad=0.1)
cbar.set_label("Temperaturveränderung/Dekade (°C)")

plt.savefig("images/spatialtrend_temp_4seasons.png", dpi=300)
plt.show()
