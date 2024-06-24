from zenml import step, pipeline, ArtifactConfig
from zenml import Model, get_step_context
import torch
import subprocess
import yaml
import os
from huggingface_hub import HfApi
from zenml import step
from zenml.client import Client
from zenml.logger import get_logger


model = Model(name="dino_classifier", description="dino classifier", version='production')

@step()
def save_model():
    model = get_step_context().model
    model_artifact = model.load_artifact('dino_classifier')
    model_path = "./gradio/model.pth"
    torch.save(model_artifact.state_dict(), model_path)
    classes = model.run_metadata["labels"].value
    print(classes)
    with open('./gradio/config.yaml', 'w') as f:
        yaml.dump({"labels": classes}, f)

@step()
def save_model_2():
    model = get_step_context().model
    model_artifact = model.load_artifact('dino_classifier')
    model_path = "./gradio_2/model.pth"
    torch.save(model_artifact.state_dict(), model_path)
    classes = model.run_metadata["labels"].value
    print(classes)
    with open('./gradio_2/config.yaml', 'w') as f:
        yaml.dump({"labels": classes}, f)

@step()
def local_deployment():

    zenml_repo_root = Client().root
    app_path = str(os.path.join(zenml_repo_root, "gradio", "app.py"))
    command = ["python", app_path]
    subprocess.Popen(command)


# Initialize logger
logger = get_logger(__name__)

@step()
def deploy_to_huggingface_image_classification(
    repo_name: str,
):
    """
    This step deploy the model to huggingface.

    Args:
        repo_name: The name of the repo to create/use on huggingface.
    """

    secret = Client().get_secret("huggingface_creds")
    assert secret, "No secret found with name 'huggingface_creds'. Please create one that includes your `username` and `token`."
    token = secret.secret_values["token"]
    api = HfApi(token=token)
    hf_repo = api.create_repo(
        repo_id=repo_name, repo_type="space", space_sdk="gradio", exist_ok=True
    )
    zenml_repo_root = Client().root
    if not zenml_repo_root:
        logger.warning(
            "You're running the `deploy_to_huggingface` step outside of a ZenML repo. "
            "Since the deployment step to huggingface is all about pushing the repo to huggingface, "
            "this step will not work outside of a ZenML repo where the gradio folder is present."
        )
        raise
    gradio_folder_path = os.path.join(zenml_repo_root, "gradio")
    space = api.upload_folder(
        folder_path=gradio_folder_path,
        repo_id=hf_repo.repo_id,
        repo_type="space",
    )
    logger.info(f"Space created: {space}")

@step()
def deploy_to_huggingface_objected_detection_classification(
    repo_name: str,
):
    """
    This step deploy the model to huggingface.

    Args:
        repo_name: The name of the repo to create/use on huggingface.
    """

    secret = Client().get_secret("huggingface_creds")
    assert secret, "No secret found with name 'huggingface_creds'. Please create one that includes your `username` and `token`."
    token = secret.secret_values["token"]
    api = HfApi(token=token)
    hf_repo = api.create_repo(
        repo_id=repo_name, repo_type="space", space_sdk="gradio", exist_ok=True
    )
    zenml_repo_root = Client().root
    if not zenml_repo_root:
        logger.warning(
            "You're running the `deploy_to_huggingface` step outside of a ZenML repo. "
            "Since the deployment step to huggingface is all about pushing the repo to huggingface, "
            "this step will not work outside of a ZenML repo where the gradio folder is present."
        )
        raise
    gradio_folder_path = os.path.join(zenml_repo_root, "gradio_2")
    space = api.upload_folder(
        folder_path=gradio_folder_path,
        repo_id=hf_repo.repo_id,
        repo_type="space",
    )
    logger.info(f"Space created: {space}")

@pipeline(enable_cache=False)
def deployment_pipeline():

    #save_model()
    #save_model_2()
    #Check everything is working fine with the app before remote deployment
    #local_deployment()
    #local_deployment_2()
    #deploy_to_huggingface_image_classification("Custom_image_classification")
    deploy_to_huggingface_objected_detection_classification("Custom_object_detection")
