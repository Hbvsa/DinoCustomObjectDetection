from zenml import pipeline
from steps.data_creation import data_creation

@pipeline(enable_cache=True)
def data_creation_pipeline():
    data_creation()
