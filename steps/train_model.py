from zenml import step, ArtifactConfig, log_model_metadata
from src.train_model import ModelTrainer
from torch import nn
from typing_extensions import Annotated
from materializers.dino_classifier_materializer import DinoMaterializer
from src.train_model import DinoVisionTransformerClassifier
@step(output_materializers={"dino_classifier": DinoMaterializer})
def train_model() -> Annotated[DinoVisionTransformerClassifier, ArtifactConfig(name="dino_classifier", is_model_artifact=True)]:

    model_trainer = ModelTrainer()
    dataset = model_trainer.return_torch_dataset()
    log_model_metadata(
        metadata={"labels": dataset.classes}
    )
    model = model_trainer.train_model(dataset)

    return model
