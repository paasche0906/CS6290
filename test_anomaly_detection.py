import pandas as pd
import numpy as np
from anomaly_detection_plot import detect_anomalies_iqr

def test_detect_anomalies_iqr_returns_indices():
    """Test that IQR function returns a numpy array of indices."""
    data = pd.Series([1, 2, 3, 100, 4, 5])
    idx = detect_anomalies_iqr(data)
    assert isinstance(idx, np.ndarray)
    assert len(idx) > 0
    assert 3 in idx  # index 3 corresponds to value 100

def test_raw_data_contains_anomaly_values():
    """Verify that a known anomaly (2024-07-16) exists in the raw CSV."""
    df = pd.read_csv('Milestone1/trump_2024_daily_history.csv')
    row = df[df['date'] == '2024-07-16']
    assert not row.empty
    # The exact price from the dataset; adjust if needed
    assert abs(row['price'].values[0] - 0.705) < 1e-6

def test_detection_stability_across_k_values():
    """Check that anomalies detected with different IQR multipliers overlap substantially."""
    df = pd.read_csv('Milestone1/trump_2024_daily_history.csv', parse_dates=['date'])
    idx_1 = detect_anomalies_iqr(df['price'], k=1.5)
    idx_2 = detect_anomalies_iqr(df['price'], k=2.0)
    common = set(idx_1) & set(idx_2)
    # Expect at least 50% overlap (adjust if needed)
    assert len(common) >= max(len(idx_1), len(idx_2)) * 0.5

def test_anomaly_alignment_with_known_events():
    """
    Verify that at least some anomalies correspond to major real-world events.
    This test uses a manually curated event list and may require updates.
    """
    # Manually collected major events (can be extended)
    major_events = {
        '2024-07-16': 'RNC convention / VP pick',
        '2024-09-10': 'Presidential debate',
        # Add more known events as needed
    }
    df = pd.read_csv('Milestone1/trump_2024_daily_history.csv', parse_dates=['date'])
    idx = detect_anomalies_iqr(df['price'])
    anomaly_dates = df.iloc[idx]['date'].dt.strftime('%Y-%m-%d').tolist()
    matched = [d for d in anomaly_dates if d in major_events]
    # Not all anomalies must match, but at least one should (adjust if necessary)
    assert len(matched) > 0, "No anomaly matches known major events"

def test_cross_source_consistency():
    """
    Threat model test: Compare with a second data source (e.g., PredictIt).
    This test assumes a file 'predictit_daily.csv' exists with columns 'date' and 'price'.
    If not available, the test can be skipped or mocked.
    """
    try:
        df_poly = pd.read_csv('Milestone1/trump_2024_daily_history.csv', parse_dates=['date'])
        df_pred = pd.read_csv('predictit_daily.csv', parse_dates=['date'])
    except FileNotFoundError:
        pytest.skip("PredictIt data file not found")

    merged = pd.merge(df_poly, df_pred, on='date', suffixes=('_poly', '_pred'))
    merged['diff'] = abs(merged['price_poly'] - merged['price_pred'])
    large_diff = merged[merged['diff'] > 0.1]  # threshold can be adjusted
    assert len(large_diff) == 0, f"Found {len(large_diff)} dates with large cross-source discrepancy"