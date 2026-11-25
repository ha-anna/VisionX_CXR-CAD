"""Multi-hot label encoding for NIH ChestX-ray14 "Finding Labels" strings."""

import numpy as np

DISEASES = (
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
)
NO_FINDING = "No Finding"


def encode_labels(finding_labels: str, diseases=DISEASES) -> np.ndarray:
    """Turn a value such as "Cardiomegaly|Effusion" into a 14-element 0/1 vector.

    "No Finding" is not a separate class: it gives all zeros.
    Unknown names raise ValueError so label typos are not silently dropped.
    """
    names = {name.strip() for name in finding_labels.split("|") if name.strip()}
    if not names:
        raise ValueError("Finding labels are empty.")
    if NO_FINDING in names:
        if len(names) > 1:
            raise ValueError(f"'{NO_FINDING}' cannot be combined with other findings.")
        return np.zeros(len(diseases), dtype=np.float32)

    unknown = names - set(diseases)
    if unknown:
        raise ValueError(f"Unknown finding labels: {sorted(unknown)}")
    return np.array([float(d in names) for d in diseases], dtype=np.float32)
