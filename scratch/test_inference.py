import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services.ml_service import get_ml_service

def test():
    req = {
        "road_type": "Tangent road with flat terrain",
        "speed_limit": 60,
        "weather": "Normal",
        "lighting": "Daylight",
        "junction": "No junction",
        "junction_ctrl": "None",
        "vehicle_type": "Automobile",
        "driver_age": "18-30",
        "urban_rural": "Unknown",
        "state": "Tangent road with flat terrain",
        "city": "Unknown"
    }

    try:
        service = get_ml_service()
        res = service.predict(req)
        print("SUCCESS!")
        print("Calibrated Probability:", res["probability"])
        print("Top 3 Features:", res["top_features"])
        print("Inference time:", res["inference_ms"], "ms")
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
