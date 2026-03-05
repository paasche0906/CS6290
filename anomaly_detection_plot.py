import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
import os

# -------------------- Data Loading --------------------
df = pd.read_csv('trump_2024_daily_history.csv', parse_dates=['date'])
df['date_only'] = df['date'].dt.date

# -------------------- Anomaly Detection Functions --------------------
def detect_anomalies_zscore(data, threshold=3):
    """Detect anomalies using Z-score."""
    z = np.abs(stats.zscore(data))
    return np.where(z > threshold)[0]

def detect_anomalies_iqr(data, k=1.5):
    """Detect anomalies using the IQR rule."""
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR
    return np.where((data < lower_bound) | (data > upper_bound))[0]

# Use IQR method (more robust for small datasets)
anomaly_idx = detect_anomalies_iqr(df['price'])
anomaly_dates = df.iloc[anomaly_idx]['date']
anomaly_prices = df.iloc[anomaly_idx]['price']

print("Detected anomaly dates and prices:")
for d, p in zip(anomaly_dates, anomaly_prices):
    print(f"{d.strftime('%Y-%m-%d')}: {p:.3f}")

# -------------------- Visualization Setup --------------------
sns.set_theme(style='whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150

fig, ax = plt.subplots(figsize=(14, 7))

# Plot the price line
ax.plot(df['date'], df['price'], linestyle='-', linewidth=2.5,
        color='#2E86AB', label='Daily Price')
ax.fill_between(df['date'], df['price'], alpha=0.2, color='#2E86AB')

# Highlight anomalies
ax.scatter(anomaly_dates, anomaly_prices, color='red', s=100,
           edgecolor='white', linewidth=1.5, zorder=5,
           label=f'Anomalies ({len(anomaly_idx)} detected)')

# Label each anomaly with date (optional, to avoid clutter you can label only a few)
for i, (d, p) in enumerate(zip(anomaly_dates, anomaly_prices)):
    ax.annotate(d.strftime('%m/%d'), xy=(d, p),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, color='darkred')

# Mark the highest price (as in your original code)
max_idx = df['price'].idxmax()
max_date = df.loc[max_idx, 'date']
max_price = df.loc[max_idx, 'price']
ax.scatter(max_date, max_price, color='#A23B72', s=80,
           edgecolor='white', linewidth=1.5, zorder=5)
ax.annotate(f'Peak: {max_price:.3f}\n{max_date.strftime("%Y-%m-%d")}',
            xy=(max_date, max_price),
            xytext=(max_date + pd.Timedelta(days=15), max_price + 0.05),
            arrowprops=dict(arrowstyle='->', color='#A23B72', lw=1.5),
            fontsize=10, color='#A23B72', weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#A23B72', lw=1))

# Election day line
election_day = pd.to_datetime('2024-11-05')
ax.axvline(election_day, color='#D95D39', linestyle='--', linewidth=1.8,
           alpha=0.8, label='Election Day (Nov 5)')
ax.text(election_day, 0.72, '🗳️ Election', rotation=90,
        verticalalignment='center', fontsize=9, color='#D95D39', weight='semibold')

# Format x-axis
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=0))
plt.xticks(rotation=45, ha='right')

ax.set_xlabel('Date', fontsize=12, weight='semibold')
ax.set_ylabel('Price (USD)', fontsize=12, weight='semibold')
ax.set_title('Trump 2024 Prediction Market Price History with Anomaly Detection',
             fontsize=16, weight='bold', pad=20)

ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
ax.grid(True, linestyle='--', alpha=0.6, which='major')
ax.grid(True, linestyle=':', alpha=0.3, which='minor')
sns.despine(ax=ax, top=True, right=True)

os.makedirs('charts', exist_ok=True)
plt.savefig('charts/trump_2024_with_anomalies.png', dpi=150)
plt.show()