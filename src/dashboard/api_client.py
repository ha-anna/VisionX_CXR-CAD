import math

import requests
from settings import API_URL, LABELS


def validate_response(data):
    if not isinstance(data, dict):
        raise ValueError("The API response must be a JSON object.")

    scores = data.get("predictions")

    if not isinstance(scores, dict) or set(scores) != set(LABELS):
        raise ValueError("The API must return scores for exactly the 14 agreed disease labels.")

    for value in scores.values():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Each disease score must be numeric.")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Disease scores must be finite values between 0 and 1.")

    detections = data.get("detected_diseases")

    if not isinstance(detections, list) or any(
        not isinstance(name, str) or name not in LABELS for name in detections
    ):
        raise ValueError("The detected disease list is missing or invalid.")

    if data.get("top1_disease") not in LABELS:
        raise ValueError("The API must identify the Top-1 disease.")

    return data


def prediction_request(raw, name, mime):
    response = requests.post(
        f"{API_URL}/predict",
        files={"file": (name, raw, mime)},
        timeout=(5, 60),
    )
    response.raise_for_status()

    return validate_response(response.json())
