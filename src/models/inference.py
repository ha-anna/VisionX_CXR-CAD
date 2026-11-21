import hashlib
import base64
import io
import json
from pathlib import Path
from threading import RLock

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.models import densenet121



def normalize_bundle_config(raw, bundle_name):
    """Expand compact config in memory; do not modify the user's config.json."""
    import copy
    p = raw.get("preprocessing", {})
    if "architecture" in raw or "bundle_version" in raw:
        return copy.deepcopy(raw)  # Full-format validation happens in the loader.
    if raw.get("model_name") != "densenet121":
        raise ValueError("Compact config supports model_name=densenet121 only.")
    if type(raw.get("num_classes")) is not int or raw["num_classes"] != 14:
        raise ValueError("Compact config requires num_classes=14.")
    if type(raw.get("in_channels")) is not int or raw["in_channels"] != 3:
        raise ValueError("Compact config requires in_channels=3.")
    if type(p.get("use_clahe")) is not bool:
        raise ValueError("preprocessing.use_clahe must be true or false.")
    size = p.get("image_size")
    if type(size) is not int or size < 32:
        raise ValueError("preprocessing.image_size must be an integer >=32.")
    unknown = set(p) - {"use_clahe", "image_size"}
    if unknown:
        raise ValueError(f"Unknown compact preprocessing keys: {sorted(unknown)}. Use the full format for explicit settings.")
    defaults = copy.deepcopy(COMPACT_DEFAULTS)
    preprocessing = {
        "grayscale": True, "clahe": p["use_clahe"], "resize": [size, size],
        "interpolation": defaults["interpolation"], "channels": "repeat_grayscale_3_times",
        "mean": defaults["mean"], "std": defaults["std"],
    }
    if p["use_clahe"]:
        preprocessing["clahe_settings"] = defaults["clahe_settings"]
    return {
        "bundle_version": 1, "model_version": bundle_name,
        "architecture": raw["model_name"], "num_classes": raw["num_classes"],
        "labels": defaults["labels"], "preprocessing": preprocessing,
        "output_activation": "sigmoid", "decision_thresholds": None,
        "score_interpretation": "Uncalibrated model scores, not disease-risk estimates.",
        "loss": copy.deepcopy(raw.get("loss", {})),
        "training_info": copy.deepcopy(raw.get("training_info", {})),
        "compact_config_defaults_applied": True,
    }


def validate_preprocessing(p):
    """Validate supported operations without fixing experiment parameter values."""
    if p.get("resize_strategy") != "direct_resize":
        raise ValueError(
            "preprocessing.resize_strategy must be 'direct_resize'."
        )
    if p.get("grayscale") is not True or p.get("channels") != "repeat_grayscale_3_times":
        raise ValueError("Supported inputs: grayscale repeated into three channels.")
    size = p.get("resize")
    if not isinstance(size, (list, tuple)) or len(size) != 2 or any(type(v) is not int or v < 32 for v in size):
        raise ValueError("resize must be [height, width], each at least 32 pixels.")
    if p.get("interpolation") not in {"INTER_AREA", "INTER_LINEAR", "INTER_CUBIC", "INTER_NEAREST"}:
        raise ValueError("Unsupported resize interpolation.")
    if type(p.get("clahe")) is not bool:
        raise ValueError("clahe must be a boolean.")
    if p["clahe"]:
        c = p.get("clahe_settings", {})
        clip = c.get("clip_limit")
        grid = c.get("tile_grid_size")
        if isinstance(clip, bool) or not isinstance(clip, (float, int)) or not np.isfinite(clip) or clip <= 0:
            raise ValueError("CLAHE clip_limit must be positive and finite.")
        if not isinstance(grid, (list, tuple)) or len(grid) != 2 or any(type(v) is not int or v <= 0 for v in grid):
            raise ValueError("CLAHE tile_grid_size must contain two positive integers.")
        if type(c.get("apply_before_resize")) is not bool:
            raise ValueError("CLAHE apply_before_resize must be a boolean.")
    for key in ["mean", "std"]:
        values = p.get(key)
        if not isinstance(values, (list, tuple)) or len(values) != 3:
            raise ValueError(f"{key} must contain three numbers.")
        if not all(isinstance(v, (float, int)) and not isinstance(v, bool) and np.isfinite(v) for v in values):
            raise ValueError(f"{key} must contain finite numbers.")
    if any(v <= 0 for v in p["std"]):
        raise ValueError("Normalization standard deviations must be positive.")
    return p


def preprocess_tensor(image, preprocessing):
    """One preprocessing implementation for training, evaluation and deployment.

    resize is [height, width]; tile_grid_size follows OpenCV's (columns, rows).
    Legacy non-CLAHE bundle configs are supported without migration.
    """
    p = validate_preprocessing(preprocessing)
    gray = np.asarray(image.convert("L"))
    clahe = None
    if p["clahe"]:
        c = p["clahe_settings"]
        clahe = cv2.createCLAHE(clipLimit=c["clip_limit"], tileGridSize=tuple(c["tile_grid_size"]))
        if c["apply_before_resize"]:
            gray = clahe.apply(gray)
    height, width = p["resize"]
    gray = cv2.resize(gray, (width, height), interpolation=getattr(cv2, p["interpolation"]))
    if clahe is not None and not c["apply_before_resize"]:
        gray = clahe.apply(gray)
    tensor = torch.from_numpy(gray.copy()).unsqueeze(0).repeat(3, 1, 1).float() / 255.0
    mean = torch.tensor(p["mean"], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(p["std"], dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


def build_model(architecture, num_classes, pretrained=False):
    """Architecture support is implemented once; saved config selects it."""
    if architecture != "densenet121":
        raise ValueError(f"Unsupported architecture: {architecture}")
    from torchvision.models import DenseNet121_Weights
    model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
    model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    return model


class VisionXPredictor:
    def __init__(self, bundle_dir, device=None):
        """Reads config.json, builds DenseNet-121, and loads the trained weights from model.pth.
        Sets the model to evaluation mode."""
        bundle = Path(bundle_dir)
        self.config = json.loads(
            (bundle / "config.json").read_text(encoding="utf-8")
        )

        required = {
            "model_version",
            "architecture",
            "num_classes",
            "labels",
            "preprocessing",
            "output_activation",
            "score_interpretation",
        }
        missing = required - self.config.keys()
        if missing:
            raise ValueError(f"Missing config fields: {', '.join(sorted(missing))}")
        if self.config.get("compact_config_defaults_applied"):
            import logging
            logging.getLogger(__name__).warning(
                "Compact model config: using fixed label order and preprocessing defaults. "
                "Verify COMPACT_DEFAULTS against training before interpreting scores. Resolved preprocessing: %s",
                self.config["preprocessing"],
            )
        if self.config.get("bundle_version", 1) not in {1, 2}:
            raise ValueError("Unsupported model bundle version.")
        self.labels = self.config["labels"]
        p = self.config["preprocessing"]
        validate_preprocessing(p)
        if (
            self.config["architecture"] != "densenet121"
            or self.config["output_activation"] != "sigmoid"
        ):
            raise ValueError("Unsupported model configuration.")
        num_classes = self.config["num_classes"]

        if type(num_classes) is not int or num_classes <= 0:
            raise ValueError("num_classes must be a positive integer.")

        if (
            not isinstance(self.labels, list)
            or not all(isinstance(label, str) and label.strip() for label in self.labels)
            or len(self.labels) != num_classes
            or len(set(self.labels)) != num_classes
        ):
            raise ValueError(
                "labels must contain exactly num_classes unique, non-empty strings."
            )
        default_device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = torch.device(device or default_device)
        self.mean = torch.tensor(p["mean"], dtype=torch.float32).reshape(3, 1, 1)
        self.std = torch.tensor(p["std"], dtype=torch.float32).reshape(3, 1, 1)
        if (
            not torch.isfinite(self.mean).all()
            or not torch.isfinite(self.std).all()
            or not (self.std > 0).all()
        ):
            raise ValueError("Invalid normalization values.")
        self.model = build_model(self.config["architecture"], self.config["num_classes"])
        weights_path = bundle / "model.pth"
        if self.config.get("bundle_version", 1) == 2:
            expected_hash = self.config.get("weights_sha256")
            if not isinstance(expected_hash, str) or hashlib.sha256(weights_path.read_bytes()).hexdigest() != expected_hash:
                raise ValueError("Model weights do not match this bundle config. Deploy the matching files together.")
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        # Last convolution in DenseNet-121; spatial size depends on configured input size.
        self.target_layer = self.model.features.denseblock4.denselayer16.conv2
        self._lock = RLock()

    def preprocess(self, image):
        """Prepares the uploaded image in the format the model expects."""
        return preprocess_tensor(image, self.config["preprocessing"]).unsqueeze(0).to(self.device)

    def predict(self, image):
        """Runs the model and returns scores for the 14 labels,
        using their order from the configuration."""
        with self._lock, torch.inference_mode():
            scores = torch.sigmoid(self.model(self.preprocess(image)))[0].cpu().tolist()
        if not np.isfinite(scores).all():
            raise RuntimeError("Non-finite model scores.")
        return dict(zip(self.labels, scores))

    def grad_cam(self, image, label):
        """Generates a colored overlay showing regions that contributed to
        a selected label's score."""
        if label not in self.labels:
            raise ValueError(f"Unknown label: {label}")

        with self._lock, torch.inference_mode(False), torch.enable_grad():
            captured = {}

            def capture(module, inputs, output):
                captured["activation"] = output

            handle = self.target_layer.register_forward_hook(capture)
            try:
                x = self.preprocess(image).requires_grad_(True)
                logits = self.model(x)
                activation = captured["activation"]
                gradient = torch.autograd.grad(logits[0, self.labels.index(label)], activation)[0]
                weights = gradient.mean(dim=(2, 3), keepdim=True)
                raw = torch.relu((weights * activation).sum(dim=1))[0].detach().cpu().numpy()
            finally:
                handle.remove()
        if not np.isfinite(raw).all():
            raise RuntimeError("Non-finite Grad-CAM values.")
        empty = float(raw.max()) <= 1e-12
        cam = np.zeros_like(raw) if empty else raw / raw.max()
        cam = cv2.resize(cam, image.size, interpolation=cv2.INTER_LINEAR)
        cam = np.clip(cam, 0, 1)
        base = np.asarray(image.convert("L").convert("RGB"), dtype=np.float32) / 255.0
        color = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        alpha = 0.45 * cam[..., None]
        overlay = Image.fromarray(np.uint8(np.clip(base * (1 - alpha) + color * alpha, 0, 1) * 255))
        return {
            "label": label,
            "heatmap": cam,
            "overlay": overlay,
            "empty": empty,
            "target_layer": "features.denseblock4.denselayer16.conv2",
        }

    def analyze(self, image, label=None):
        """Combines predictions and Grad-CAM, then packages the results for the API."""
        scores = self.predict(image)
        top_label = max(scores, key=scores.get)
        explanation = self.grad_cam(image, label or top_label)
        buffer = io.BytesIO()
        explanation["overlay"].save(buffer, format="PNG")
        return {
            "model_version": self.config["model_version"],
            "scores": scores,
            "score_interpretation": self.config["score_interpretation"],
            "top_label": top_label,
            "gradcam_label": explanation["label"],
            "gradcam_empty": explanation["empty"],
            "gradcam_target_layer": explanation["target_layer"],
            "gradcam_overlay_png_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        }
