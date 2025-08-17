import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from rasterio.features import geometry_mask
from scipy.stats import t

# Pfade zu den Dateien
temp_file_path = "data/era5_temp_1950_2024.grib"
runoff_file_path = "data/era5_runoff_1950_2024.grib"
vector_path = "qgis_data/gadm41_UZB.gpkg"

# 1. Temperaturdaten laden
temp_data = xr.open_dataset(temp_file_path, engine="cfgrib")
temperature = temp_data['t2m'] - 273.15  # Umwandlung von Kelvin nach Celsius
temperature = temperature.sel(time=slice("1991-01-01", "2020-12-01"))

# 2. Abflussdaten laden
ds = xr.open_dataset(runoff_file_path, engine="cfgrib", filter_by_keys={'stepType': 'avgad'})
ds2 = xr.open_dataset(runoff_file_path, engine="cfgrib", filter_by_keys={'stepType': 'avgas'})

# Kombinieren der Zeitreihen
runoff_data = xr.concat([ds, ds2], dim="time").sortby('time')
runoff_mm = runoff_data['ro'] * 1000  # Abfluss in mm umrechnen
runoff_mm = runoff_mm.sel(time=slice("1991-01-01", "2020-12-01"))

# 3. Geometrie für Usbekistan laden
gdf = gpd.read_file(vector_path, layer="ADM_ADM_0")
usbekistan_geometry = gdf.geometry.union_all()

# 4. Maske erstellen
lat, lon = temperature.latitude.values, temperature.longitude.values
transform = temperature.rio.transform()

mask = geometry_mask(
    [usbekistan_geometry],
    transform=transform,
    invert=True,
    out_shape=(len(lat), len(lon))
)

mask_da = xr.DataArray(mask, coords={"latitude": lat, "longitude": lon}, dims=["latitude", "longitude"])
temperature = temperature.where(mask_da)
runoff_mm = runoff_mm.where(mask_da)

# 5. Definition der Jahreszeiten
seasons = {
    "Dezember – Februar": [12, 1, 2],
    "März – Mai": [3, 4, 5],
    "Juni – August": [6, 7, 8],
    "September – November": [9, 10, 11]
}

# 6. Erstellung der Grafik
fig, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={"projection": ccrs.Mercator()})
fig.suptitle("Räumliche Korrelation & Signifikanz zwischen Temperatur und Abfluss (1991-2020)", fontsize=16, fontweight="bold")

# Dynamische Levels für die Korrelation
levels = np.linspace(-1, 1, num=21)  # Korrelation von -1 bis 1

for ax, (season_name, months) in zip(axes.flat, seasons.items()):
    temp_season = temperature.sel(time=temperature['time.month'].isin(months))
    runoff_season = runoff_mm.sel(time=runoff_mm['time.month'].isin(months))

    # 6a. Korrelation berechnen
    correlation_map = xr.corr(temp_season, runoff_season, dim="time")

    # 6b. Berechnung der Signifikanz
    n = temp_season.time.size
    t_stat_map = correlation_map * np.sqrt(n - 2) / np.sqrt(1 - correlation_map**2)
    p_map = xr.apply_ufunc(lambda t_stat: 2 * (1 - t.cdf(np.abs(t_stat), df=n - 2)), t_stat_map, vectorize=True)
    significant_map = p_map < 0.05  # Signifikanzschwelle p < 0.05

    # 7. Plot der Korrelation
    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
    
    correlation_plot = ax.contourf(
        correlation_map.longitude,
        correlation_map.latitude,
        correlation_map,
        levels=levels,
        cmap="coolwarm",
        transform=ccrs.PlateCarree()
    )

    # 8. Signifikante Bereiche mit Schraffur markieren
    significance_plot = ax.contourf(
        correlation_map.longitude,
        correlation_map.latitude,
        significant_map,
        levels=[0, 0.5, 1],  # Binäre Maske: 0 = nicht signifikant, 1 = signifikant
        colors="none",
        hatches=["", "..."],  # Schraffur für signifikante Bereiche
        transform=ccrs.PlateCarree()
    )

    ax.set_title(season_name, fontsize=14, fontweight="bold")

# 9. Gemeinsame Farbleiste (Legende)
cbar = fig.colorbar(correlation_plot, ax=axes, orientation="horizontal", fraction=0.05, pad=0.1)
cbar.set_label("Pearson-Korrelation (r)")

plt.tight_layout(rect=[0, 0.15, 1, 0.95])  # Platz für Haupttitel lassen
plt.savefig("images/spatial_correlation_temp_runoff_4seasons_ttest.png", dpi=300)
plt.show()
