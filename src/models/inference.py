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


class VisionXPredictor:
    def __init__(self, bundle_dir, device=None):
        """Reads config.json, builds DenseNet-121, and loads the trained weights from model.pth.
        Sets the model to evaluation mode."""
        bundle = Path(bundle_dir)
        self.config = json.loads((bundle / "config.json").read_text())
        self.labels = self.config["labels"]
        p = self.config["preprocessing"]
        expected = {
            "grayscale": True,
            "clahe": False,
            "resize": [224, 224],
            "interpolation": "INTER_AREA",
            "channels": "repeat_grayscale_3_times",
        }
        if (
            self.config["architecture"] != "densenet121"
            or self.config["output_activation"] != "sigmoid"
        ):
            raise ValueError("Unsupported model configuration.")
        if (
            len(self.labels) != 14
            or len(set(self.labels)) != 14
            or self.config["num_classes"] != 14
        ):
            raise ValueError("Expected 14 unique labels.")
        if any(p.get(k) != v for k, v in expected.items()):
            raise ValueError("This module supports the exported baseline preprocessing only.")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.mean = torch.tensor(p["mean"], dtype=torch.float32).reshape(3, 1, 1)
        self.std = torch.tensor(p["std"], dtype=torch.float32).reshape(3, 1, 1)
        if (
            not torch.isfinite(self.mean).all()
            or not torch.isfinite(self.std).all()
            or not (self.std > 0).all()
        ):
            raise ValueError("Invalid normalization values.")
        self.model = densenet121(weights=None)
        self.model.classifier = torch.nn.Linear(self.model.classifier.in_features, 14)
        state = torch.load(bundle / "model.pth", map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        # Last convolution in DenseNet-121. Spatial maps are 7x7 for 224x224 input.
        self.target_layer = self.model.features.denseblock4.denselayer16.conv2
        self._lock = RLock()

    def preprocess(self, image):
        """Prepares the uploaded image in the format the model expects."""
        gray = np.asarray(image.convert("L"))
        gray = cv2.resize(gray, (224, 224), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(gray.copy()).unsqueeze(0).repeat(3, 1, 1).float() / 255.0
        return ((tensor - self.mean) / self.std).unsqueeze(0).to(self.device)

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
