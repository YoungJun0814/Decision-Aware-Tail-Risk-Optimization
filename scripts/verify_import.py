import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.models import get_model
    print("SUCCESS: src.models imported correctly.")
except ImportError as e:
    print(f"FAIL: {e}")
except Exception as e:
    print(f"FAIL: {e}")
