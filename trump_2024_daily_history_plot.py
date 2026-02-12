import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Load and prepare data
df = pd.read_csv('trump_2024_daily_history.csv', parse_dates=['date'])
df['date_only'] = df['date'].dt.date

sns.set_theme(style='whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150

# Create figure and axis
fig, ax = plt.subplots(figsize=(14, 7))

# Plot the price line with gradient color effect

line = ax.plot(df['date'], df['price'], 
               linestyle='-', linewidth=2.5, 
               color='#2E86AB', label='Daily Price')

# Fill below the line with light blue
ax.fill_between(df['date'], df['price'], alpha=0.2, color='#2E86AB')

# Annotate the highest price point
max_idx = df['price'].idxmax()
max_date = df.loc[max_idx, 'date']
max_price = df.loc[max_idx, 'price']

ax.scatter(max_date, max_price, color='#A23B72', s=80, 
           edgecolor='white', linewidth=1.5, zorder=5, 
           label=f'Peak: {max_price:.3f}')

ax.annotate(f'Peak: {max_price:.3f}\n{max_date.strftime("%Y-%m-%d")}',
            xy=(max_date, max_price),
            xytext=(max_date + pd.Timedelta(days=15), max_price + 0.05),
            arrowprops=dict(arrowstyle='->', color='#A23B72', lw=1.5),
            fontsize=10, color='#A23B72', weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#A23B72', lw=1))


election_day = pd.to_datetime('2024-11-05')
ax.axvline(election_day, color='#D95D39', linestyle='--', linewidth=1.8, 
           alpha=0.8, label='Election Day (Nov 5)')

# Add a subtle text annotation for Election Day
ax.text(election_day, 0.72, '🗳️ Election', 
        rotation=90, verticalalignment='center',
        fontsize=9, color='#D95D39', weight='semibold')

# Format x-axis dates
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # e.g., Jan 2024
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=0))  # Mondays
plt.xticks(rotation=45, ha='right')

# Set labels, title, and legend
ax.set_xlabel('Date', fontsize=12, weight='semibold')
ax.set_ylabel('Price (USD)', fontsize=12, weight='semibold')
ax.set_title('Trump 2024 Prediction Market Price History', 
             fontsize=16, weight='bold', pad=20)

# Legend with nicer placement
ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)

ax.grid(True, linestyle='--', alpha=0.6, which='major')
ax.grid(True, linestyle=':', alpha=0.3, which='minor')
sns.despine(ax=ax, top=True, right=True)

plt.tight_layout()

plt.show()