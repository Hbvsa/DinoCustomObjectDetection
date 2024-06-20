import os
import tempfile
from typing import Any, ClassVar, Type

from src.train_model import DinoVisionTransformerClassifier
from zenml.integrations.pytorch.materializers.pytorch_module_materializer import (
    PyTorchModuleMaterializer,
)
from zenml.io import fileio

DEFAULT_FILENAME = "obj.pt"
import torch

class DinoMaterializer(PyTorchModuleMaterializer):
    """Base class for ultralytics YOLO models."""

    FILENAME: ClassVar[str] = DEFAULT_FILENAME
    SKIP_REGISTRATION: ClassVar[bool] = True
    ASSOCIATED_TYPES = (DinoVisionTransformerClassifier,)

    def load(self, data_type: Type[Any]) -> DinoVisionTransformerClassifier:
        """Reads a ultralytics YOLO model from a serialized JSON file.

        Args:
            data_type: A ultralytics YOLO type.

        Returns:
            A ultralytics YOLO object.
        """
        filepath = os.path.join(self.uri, DEFAULT_FILENAME)

        # Create a temporary folder
        with tempfile.TemporaryDirectory(prefix="zenml-temp-") as temp_dir:
            temp_file = os.path.join(str(temp_dir), DEFAULT_FILENAME)

            # Copy from artifact store to temporary file
            fileio.copy(filepath, temp_file)
            state_dict = torch.load(temp_file)
            model = DinoVisionTransformerClassifier()
            model.load_state_dict(state_dict)
            return model

    def save(self, model: DinoVisionTransformerClassifier) -> None:
        """Creates a JSON serialization for a ultralytics YOLO model.

        Args:
            model: A ultralytics YOLO model.
        """
        filepath = os.path.join(self.uri, DEFAULT_FILENAME)

        # Make a temporary phantom artifact
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            torch.save(model.state_dict(), f.name)
            f.close()
            fileio.copy(f.name, filepath)