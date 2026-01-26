import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import ASSET_TICKERS

print(f"Current Asset Tickers: {ASSET_TICKERS}")

if 'SH' not in ASSET_TICKERS and 'BIL' in ASSET_TICKERS:
    print("SUCCESS: SH removed, BIL added.")
else:
    print("FAIL: Asset list incorrect.")
