import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# ----------------------------
# 1. Load and prepare data
# ----------------------------
df = pd.read_csv('FINAL_trump_2024_ready_for_viz.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

# Basic data integrity check
assert not df['price'].isnull().any(), "Missing price values detected"
print(f"Data loaded: {len(df)} rows from {df['date'].min().date()} to {df['date'].max().date()}")

# ----------------------------
# 2. Anomaly detection (IQR)
# ----------------------------
def detect_anomalies_iqr(series, multiplier=1.5):
    """Return boolean mask for anomalies using IQR rule."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    return (series < lower_bound) | (series > upper_bound)

anomaly_mask = detect_anomalies_iqr(df['price'])
df['anomaly'] = anomaly_mask

# Classify anomalies: spike (above upper bound) or drop (below lower bound)
q1 = df['price'].quantile(0.25)
q3 = df['price'].quantile(0.75)
iqr = q3 - q1
upper = q3 + 1.5 * iqr
lower = q1 - 1.5 * iqr

df['anomaly_type'] = 'normal'
df.loc[anomaly_mask & (df['price'] > upper), 'anomaly_type'] = 'spike'
df.loc[anomaly_mask & (df['price'] < lower), 'anomaly_type'] = 'drop'

# ----------------------------
# 3. Define key events for annotation
# ----------------------------
events = {
    '2024-07-16': ('Peak: $0.705\n(RNC begins)', 'red'),
    '2024-11-05': ('Election Day', 'blue'),
    '2024-07-14': ('Assassination attempt', 'orange'),
    '2024-10-30': ('Pre-election surge', 'green'),
    '2024-11-02': ('Sharp drop', 'purple')
}

# Convert event dates to datetime for plotting
event_dates = {datetime.strptime(d, '%Y-%m-%d'): (label, color) for d, (label, color) in events.items()}

# ----------------------------
# 4. Generate final chart
# ----------------------------
fig, ax = plt.subplots(figsize=(14, 7))

# Plot price line
ax.plot(df['date'], df['price'], color='steelblue', linewidth=1.5, label='Daily Price')

# Fill area under the curve
ax.fill_between(df['date'], df['price'], alpha=0.2, color='steelblue')

# Mark anomalies
spikes = df[df['anomaly_type'] == 'spike']
drops = df[df['anomaly_type'] == 'drop']
ax.scatter(spikes['date'], spikes['price'], color='red', s=80, marker='o', edgecolors='darkred', label='Spike (IQR outlier)')
ax.scatter(drops['date'], drops['price'], color='green', s=80, marker='v', edgecolors='darkgreen', label='Drop (IQR outlier)')

# Annotate events
for date, (label, color) in event_dates.items():
    # Find the price on that date (or closest)
    price_row = df[df['date'] == date]
    if not price_row.empty:
        y = price_row['price'].values[0]
        ax.annotate(label, xy=(date, y), xytext=(date, y + 0.03),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5),
                    fontsize=9, color=color, ha='center')

# Add vertical line for election day
election_date = datetime(2024, 11, 5)
ax.axvline(x=election_date, linestyle='--', color='gray', alpha=0.7, label='Election Day')

# Formatting
ax.set_title('Trump 2024 Prediction Market Price History\nwith Detected Anomalies and Key Events', fontsize=14)
ax.set_xlabel('Date')
ax.set_ylabel('Price (USD)')
ax.legend(loc='upper left')
ax.grid(True, linestyle='--', alpha=0.5)

# Improve date formatting
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('trump_2024_final_report.png', dpi=300)
print("Chart saved as 'trump_2024_final_report.png'")
plt.show()

# ----------------------------
# 5. Output anomaly summary table
# ----------------------------
anomaly_df = df[df['anomaly']].copy()
anomaly_df = anomaly_df[['date', 'price', 'anomaly_type']].sort_values('date')

print("\n" + "="*60)
print("Detected Anomalies:")
print(anomaly_df.to_string(index=False))

# ----------------------------
# 6. Validation: Stability across IQR multipliers
# ----------------------------
print("\n" + "="*60)
print("Validation: Stability across IQR multipliers")
for k in [1.5, 2.0]:
    mask_k = detect_anomalies_iqr(df['price'], multiplier=k)
    count = mask_k.sum()
    print(f"IQR multiplier = {k}: {count} anomalies detected")

# Check that the July 16 peak is always an anomaly
july16_peak = df[df['date'] == '2024-07-16']['price'].values[0]
print(f"\nValidation: July 16 price ({july16_peak}) is an outlier?")
for k in [1.5, 2.0]:
    mask_k = detect_anomalies_iqr(df['price'], multiplier=k)
    is_outlier = mask_k[df['date'] == '2024-07-16'].values[0]
    print(f"  multiplier={k}: {is_outlier}")

# ----------------------------
# 7. Validation: Event correlation (manual check in code)
# ----------------------------
print("\n" + "="*60)
print("Validation: Event correlation (manual)")
print("List of anomalies with associated events:")
for idx, row in anomaly_df.iterrows():
    date_str = row['date'].strftime('%Y-%m-%d')
    if date_str in events:
        print(f"  {date_str} (${row['price']}) - {events[date_str][0]}")
    else:
        # Check if any event is within +/- 2 days (simplified)
        found = False
        for ev_date_str, (ev_label, _) in events.items():
            ev_date = datetime.strptime(ev_date_str, '%Y-%m-%d')
            if abs((row['date'] - ev_date).days) <= 2:
                print(f"  {date_str} (${row['price']}) - close to: {ev_label} (within 2 days)")
                found = True
                break
        if not found:
            print(f"  {date_str} (${row['price']}) - no direct event correlation")