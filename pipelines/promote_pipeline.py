from zenml import step, pipeline, ArtifactConfig
from src.train_model import ModelTrainer, DinoVisionTransformerClassifier
from typing_extensions import Annotated
from zenml import Model, get_step_context, get_pipeline_context
from torch import nn
from steps.train_model import train_model
from src.train_model import ModelTrainer
from materializers.dino_classifier_materializer import DinoMaterializer

model = Model(name="dino_classifier", description="dino classifier", version='latest')

@step(output_materializers={"dino_classifier": DinoMaterializer})
def load_model() -> Annotated[DinoVisionTransformerClassifier, ArtifactConfig(name="dino_classifier", is_model_artifact=True)]:
    model = get_step_context().model
    model_artifact = model.load_artifact('dino_classifier')
    return model_artifact

@step
def evaluate_model(dino_classifier: DinoVisionTransformerClassifier) -> Annotated[int, "accuracy"]:
    model_trainer = ModelTrainer()
    accuracy = model_trainer.evaluate_model(dino_classifier)
    return accuracy

@step
def promote_model(accuracy: int):
    if accuracy > 90:
        model = get_step_context().model
        model.set_stage("production", force=True)
        print(f"Current model promoted with accuracy {accuracy}")
    else:
        print(f"Current model not promoted with accuracy {accuracy}")
        
@step
def save_model(dino_classifier: DinoVisionTransformerClassifier):

    model_path = "./saved_model/model.pth"
    torch.save(dino_classifier.state_dict(), model_path)
    classes = model.run_metadata["labels"].value
    print(classes)
    with open('./saved_model/config.yaml', 'w') as f:
        yaml.dump({"labels": classes}, f)
        
@pipeline(enable_cache=False, model=model)
def promote_pipeline():
    dino_classifier = load_model()
    accuracy = evaluate_model(dino_classifier)
    promote_model(accuracy)
    save_model(dino_classifier)


