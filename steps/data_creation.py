from zenml import step
from src.generate_dataset import DatasetGenerator

@step
def data_creation():

    datasetGenerator = DatasetGenerator(
        video_path='videos',
        text_prompt='human face',
        destination_folder='dataset'
    )
    datasetGenerator.generate_full_dataset()

