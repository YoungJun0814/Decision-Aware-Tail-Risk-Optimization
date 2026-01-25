from src.data_loader import ASSET_TICKERS, INVERSE_ETF_INDEX

print(f"Current Asset Tickers: {ASSET_TICKERS}")
print(f"Inverse ETF Index: {INVERSE_ETF_INDEX}")

if 'SH' not in ASSET_TICKERS and 'BIL' in ASSET_TICKERS:
    print("SUCCESS: SH removed, BIL added.")
else:
    print("FAIL: Asset list incorrect.")
