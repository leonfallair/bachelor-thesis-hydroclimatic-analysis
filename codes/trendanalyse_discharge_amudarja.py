import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Dateipfade der Stationen
file_paths = {
    "Kelif": "discharge_data_monthly/Amudarya_Kelif_1990-2018_cleaned.csv",
    "Tuyamuyun": "discharge_data_monthly/Amudarya_downstream_Tuyamuyun_1976-2016_cleaned.csv",
    "Chatly": "discharge_data_monthly_GRDC/Chatly_31_73_interpolated.csv",
    "Samanbay": "discharge_data_monthly/Amudarya_samanbay_1990-2018_cleaned.csv",
}

# Koordinaten und Höhen der Stationen
station_info = {
    "Kelif": {"lat": 37.32, "lon": 66.29, "alt": 266},
    "Tuyamuyun": {"lat": 41.23, "lon": 61.37, "alt": 118},
    "Chatly": {"lat": 42.28, "lon": 59.7, "alt": 72},
    "Samanbay": {"lat": 42.36, "lon": 59.0, "alt": 71},
}

# Funktion zum Laden und Verarbeiten der Daten
def process_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data['Year'] = data['Year'].astype(int)
    data['Value'] = data['Value'].astype(float)

    yearly_data = data.groupby('Year')['Value'].mean().reset_index()

    # Fehlende Jahre ergänzen
    all_years = pd.DataFrame({'Year': range(yearly_data['Year'].min(), yearly_data['Year'].max() + 1)})
    yearly_data = all_years.merge(yearly_data, on='Year', how='left')

    # Fehlende Werte interpolieren
    yearly_data['Value'] = yearly_data['Value'].interpolate(method='linear')

    # Lineare Regression
    X = yearly_data['Year'].values
    y = yearly_data['Value'].values
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const)
    results = model.fit()

    # Werte aus der Regression
    trend = results.predict(X_with_const)
    slope = results.params[1]  # Steigung
    intercept = results.params[0]  # Achsenabschnitt
    r_squared = results.rsquared  # Bestimmtheitsmaß
    p_value = results.pvalues[1]  # p-Wert für die Steigung

    # Gleitender Durchschnitt (5 Jahre)
    yearly_data["SMA_5"] = yearly_data["Value"].rolling(window=5, center=True).mean()

    return yearly_data, trend, slope, r_squared, p_value

# Plot-Vorbereitung
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()  # Array in 1D umwandeln

# Farben für die Trends & Linien
colors = {
    "Jährlicher Durchschnitt": "gray",
    "Linearer Trend": "red",
    "5-Jahres-Smoothing": "green"
}

# Alle Stationen durchlaufen
for idx, (station, path) in enumerate(file_paths.items()):
    yearly_data, trend, slope, r_squared, p_value = process_data(path)
    ax = axs[idx]

    # Daten plotten
    ax.plot(yearly_data["Year"], yearly_data["Value"], label="Jährlicher Durchschnitt", color=colors["Jährlicher Durchschnitt"], alpha=0.6)
    ax.plot(yearly_data["Year"], trend, label="Linearer Trend", color=colors["Linearer Trend"], linestyle="--")
    ax.plot(yearly_data["Year"], yearly_data["SMA_5"], label="5-Jahres-Smoothing", color=colors["5-Jahres-Smoothing"], linestyle="-.")

    # Station & Geodaten
    lat, lon, alt = station_info[station].values()
    ax.annotate(
        f"{station}\n{lat}°N, {lon}°E\n{alt} m ü. NN",
        xy=(0.5, 0.98), xycoords="axes fraction",
        fontsize=12, va="top", ha="center",
        bbox=dict(facecolor="white", alpha=0.3, edgecolor="black", boxstyle="round,pad=0.3")
    )

    # Regressionstext
    ax.text(
        0.02, 0.02,
        f"Steigung: {slope:.2f} m³/s/Jahr\n$R^2$ = {r_squared:.3f}\np = {p_value:.4f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.6),
    )

    ax.set_xlabel("Jahr")
    ax.set_ylabel("Durchfluss (m³/s)")
    ax.grid(alpha=0.4)

# Gesamttitel & Legende
fig.suptitle("Jährlicher Abfluss des Amudarja an vier Stationen mit Trendlinien", fontsize=17, fontweight="bold")
fig.legend(["Jährlicher Durchschnitt", "Linearer Trend", "5-Jahres-Smoothing"],
           loc="lower center", ncol=3, fontsize=12, frameon=True, bbox_to_anchor=(0.5, 0.02))

plt.tight_layout(rect=[0, 0.05, 1, 0.96])  # Platz für Titel und Legende lassen
plt.savefig("images/discharge_amudarja.png", dpi=300)
plt.show()
