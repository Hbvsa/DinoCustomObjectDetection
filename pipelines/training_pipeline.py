from zenml import pipeline
from zenml import Model
from steps.train_model import train_model
model = Model(name="dino_classifier", description="dino classifier")

@pipeline(enable_cache=False, model=model)
def training_pipeline():
    dino_classifier = train_model()
