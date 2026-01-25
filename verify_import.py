try:
    from src.models import get_model
    print("SUCCESS: src.models imported correctly.")
except ImportError as e:
    print(f"FAIL: {e}")
except Exception as e:
    print(f"FAIL: {e}")
