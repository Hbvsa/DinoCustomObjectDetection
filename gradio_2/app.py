import subprocess

def run_commands():
    commands = [
        "apt-get update",
        "apt-get install -y libgl1",
        "git clone https://github.com/IDEA-Research/GroundingDINO.git",
        "pip install -e ./GroundingDINO",
        "cd GroundingDINO",
        "mkdir weights",
        "wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
        "cd .."
    ]

    for command in commands:
        try:
            print(f"Running command: {command}")
            result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(result.stdout.decode())
        except subprocess.CalledProcessError as e:
            print(f"Command '{command}' failed with error: {e.stderr.decode()}")

# Call the function to run the commands

if __name__ == "__main__":
    run_commands()

    from typing import List
    from Utils import get_video_properties
    from GroundingDINO.groundingdino.util.inference import load_model, predict
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    import GroundingDINO.groundingdino.datasets.transforms as T
    from torchvision.ops import box_convert
    from torchvision import transforms
    from torch import nn
    from os.path import dirname, abspath
    import yaml
    import supervision as sv
    import gradio as gr
    import spaces

    class DinoVisionTransformerClassifier(nn.Module):
        def __init__(self):
            super(DinoVisionTransformerClassifier, self).__init__()
            self.transformer = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
            self.classifier = nn.Sequential(nn.Linear(384, 256), nn.ReLU(), nn.Linear(256, 2))

        def forward(self, x):
            x = self.transformer(x)
            x = self.transformer.norm(x)
            x = self.classifier(x)
            return x


    class ImageClassifier:

        def __init__(self):
            with open(f"{dirname(abspath(__file__))}/config.yaml", 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
                labels = config["labels"]

            self.labels = labels
            self.dino = DinoVisionTransformerClassifier()
            model_path = f"{dirname(abspath(__file__))}/model.pth"
            state_dict = torch.load(model_path)
            self.dino.load_state_dict(state_dict)

        def preprocess(self, image: np.ndarray) -> torch.Tensor:
            data_transforms = {
                "test": transforms.Compose(
                    [
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
                    ]
                )
            }
            image_pillow = Image.fromarray(image)
            img_transformed = data_transforms['test'](image_pillow)

            return img_transformed

        def predict(self, image):
            image = self.preprocess(image)
            image = image.unsqueeze(0)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.dino.to(device)
            self.dino.eval()
            with torch.no_grad():
                output = self.dino(image.to(device))

            logit, predicted = torch.max(output.data, 1)
            return self.labels[predicted[0].item()], logit[0].item()


    class VideoObjectDetection:

        def __init__(self,
                     text_prompt: str
                     ):

            self.text_prompt = text_prompt

        def crop(self, frame, boxes):

            h, w, _ = frame.shape
            boxes = boxes * torch.Tensor([w, h, w, h])
            xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            min_col, min_row, max_col, max_row = map(int, xyxy[0])
            crop_image = frame[min_row:max_row, min_col:max_col, :]

            return crop_image

        def annotate(self,
                     image_source: np.ndarray,
                     boxes: torch.Tensor,
                     logits: torch.Tensor,
                     phrases: List[str],
                     frame_rgb: np.ndarray,
                     classifier) -> np.ndarray:

            h, w, _ = image_source.shape
            boxes = boxes * torch.Tensor([w, h, w, h])
            xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            detections = sv.Detections(xyxy=xyxy)
            print(xyxy.shape)
            custom_labels = []
            custom_logits = []

            for box in xyxy:
                min_col, min_row, max_col, max_row = map(int, box)
                crop_image = frame_rgb[min_row:max_row, min_col:max_col, :]
                label, logit = classifier.predict(crop_image)
                print()
                if logit >= 1:
                    custom_labels.append(label)
                    custom_logits.append(logit)
                else:
                    custom_labels.append('unknown human face')
                    custom_logits.append(logit)

            labels = [
                f"{phrase} {logit:.2f}"
                for phrase, logit
                in zip(custom_labels, custom_logits)
            ]

            box_annotator = sv.BoxAnnotator()
            annotated_frame = box_annotator.annotate(scene=image_source, detections=detections, labels=labels)
            return annotated_frame

        def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
            transform = T.Compose(
                [
                    T.RandomResize([800], max_size=1333),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

            image_pillow = Image.fromarray(image)
            image_transformed, _ = transform(image_pillow, None)
            return image_transformed

        def generate_video(self, video_path) -> None:

            # Load model, set up variables and get video properties
            cap, fps, width, height, fourcc = get_video_properties(video_path)
            model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                               "GroundingDINO/weights/groundingdino_swint_ogc.pth")
            predictor = ImageClassifier()
            TEXT_PROMPT = self.text_prompt
            BOX_TRESHOLD = 0.6
            TEXT_TRESHOLD = 0.6

            # Read video frames, crop image based on text prompt object detection and generate dataset_train
            import time
            frame_count = 0
            delay = 1 / fps  # Delay in seconds between frames
            while cap.isOpened():
                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    break

                if cv2.waitKey(1) & 0xff == ord('q'):
                    break

                # Convert bgr frame to rgb frame to image to torch tensor transformed
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_transformed = self.preprocess_image(frame_rgb)

                boxes, logits, phrases = predict(
                    model=model,
                    image=image_transformed,
                    caption=TEXT_PROMPT,
                    box_threshold=BOX_TRESHOLD,
                    text_threshold=TEXT_TRESHOLD
                )

                # Get boxes
                if boxes.size()[0] > 0:
                    annotated_frame = self.annotate(image_source=frame, boxes=boxes, logits=logits,
                                                    phrases=phrases, frame_rgb=frame_rgb, classifier=predictor)
                    # cv2.imshow('Object detection', annotated_frame)
                    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                yield frame_rgb
                elapsed_time = time.time() - start_time
                time_to_wait = max(delay - elapsed_time, 0)
                time.sleep(time_to_wait)

            frame_count += 1


    @spaces.GPU(duration=200)
    def video_object_classification_pipeline():
        video_annotator = VideoObjectDetection(
            text_prompt='human face')

        with gr.Blocks() as iface:
            video_input = gr.Video(label="Upload Video")
            run_button = gr.Button("Start Processing")
            output_image = gr.Image(label="Classified video")
            run_button.click(fn=video_annotator.generate_video, inputs=video_input,
                             outputs=output_image)

        iface.launch(share=False, debug=True)

    print("Só me falta a GPU")
    video_object_classification_pipeline()