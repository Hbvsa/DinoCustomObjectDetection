# Project overview

![](Assets/overview.png)
The project starts by creating dataset of images from cropped objects which are detected using GroundingDINO model with a text prompt. The dataset is used to train a image classifier with our custom labels for the objects detected. Finally the purpose of the project is to do an inference composed of the GroundingDINO object detection model with the same initial text prompt combined with the trained classifier. This allows the models to detect and classify the objects observed during training with new custom labels to annotate the frames of a video.
Future work: use the current project to automatically track objects with custom labels to apply stable diffusion for video editing
# Setup and running instructions
## Setup for project


Install the requirements

```
pip install --no-cache-dir -r requirements.txt
```
Get the GroundingDINO model to generate our object detections with text prompts.
For that we will clone and setup using the project from [https://github.com/IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO).
Run the following commands.

```
git clone https://github.com/IDEA-Research/GroundingDINO.git
pip install -e ./GroundingDINO
mkdir ./GroundingDINO/weights
wget -P ./GroundingDINO/weights -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```
##  Dataset creation

Remove the example videos and add your videos to the videos folder.

The videos need to be named with the format Class_index.mp4 (the video format can be different from mp4 as long as opencv-python library can read it), for example:
- jason_0.mp4
- jason_1.mp4
- lacy_0.mp4
- lacy_1.mp4
- jason_test.mp4
-  lacy_test.mp4

The code will assume the Class is the name you want to give to the object/animal/person detected in that video. To test the accuracy of the model on a test video please provide videos with the format Class_test.mp4. 

In the videos please make sure the prompt you will use to capture the desired object/animal/person will not capture other object/animal/person with the same general designation. For example, for different person scenarios, use videos of two people but in each video only the desired person appears. If not possible, remove manually the undesired images after creating the datasets.  *This is only necessary during the training/test dataset creation, obviously during inference the point is to have multiple possible object/person/animals be classified with the custom desired labels.*

One general advice is that it is better to provided small videos for example 2 videos of 3 minutes for each class instead of a single long video where the person is under the same lightning and scenario. The test video should also be in a different scenario from the training videos to ensure the testing accuracy is meaningful. 

In our example with the two classes 'jason' and 'lacy' we provided two 3 minute videos for each class for training and one 3 minute video for each class for testing.

![](Assets/data_creation.png)

To change the text prompt go to steps/data_creation.py and directly change it there. The default is "human face". You can also change the videos path if you are not using the default videos folder in the project repo. Do not change the destination_folder since the training pipeline is expecting this path.

The generated dataset will crop the object every 3 seconds, this is done to accelerate the process, if you want more images produced per video go to src/generate_dataset and in the generate_class_dataset method change the seconds in the if condition frame_count % (60 * seconds). It is also in this method that you can change the box and text thresholds if you wish too. The default used in the code is 0.6.


![](Assets/frames_crop.png)

Now you can run the run_data_creation.py to create the dataset.
```
python run_data_creation.py
```
Ignore the future/user warnings.
The console should be outputting the directories for the class folders being created.
If the class folders already have files from previous videos the code is able to handle this and add the new data. In the case of a completely new class dataset it will say no files found.

![](Assets/folder_creation.png)

Although be careful to not repeat samples from the same videos.
It should end with this message
![](Assets/data_creation_finish.png)
## Model training
Once you have your datasets you can proceed to train the classifier model. When reading the dataset images the training dataset will double in size by applying horizontal flips to the images.

```
python run_training.py
```


![](Assets/train_step.png)
During the training script it will output accuracy metrics. In our simple example with only two classes we ran the model for a single epoch and it achieved 99% accuracy on the test videos.

The training code is just a regular pytorch model training. To change the number of epochs, optimizer or anything else feel free to change the code at src/train_model.

After running the training pipeline the run_training.py will also trigger the promote pipeline which will reload the model and evaluate again. By default the required accuracy for the model to be promoted is 90. You can increase or lower this threshold by changing it directly on the pipelines/promote_pipeline.py file in the promote_model function.
![](Assets/promote.png)
The promotion is necessary for the deployment pipeline to have access to the model.
The model is now saved at saved_model along with a yaml file describing the classes.
## Inference App

### Local setup

One slight problem is the incompatibility of zenml with gradio since they both require different pydantic versions. So when using the app you need to run pip install gradio.

```
pip install gradio
python app.py
```
![](Assets/url.png)
By running the app.py it will launch a gradio app click the link to open the app in your browser.
![](Assets/gradio.png)
After clicking and uploading a video click Start Processing (ignore the fact the uploaded video is not displaying). After you click processing you should see the video with the object detection box and their classes. If neither the labels achieves a logit above 0 it will say 'uknown' instead.
![](Assets/detection.png)

Switch back to working with zenml if you wish by running to get the right pydantic version.
```
pip install zenml
```
