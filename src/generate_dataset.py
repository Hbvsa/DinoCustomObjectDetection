import sys
import os

# Add the relative path to the system path
current_dir = os.path.dirname(os.path.abspath(__name__))
project_root = os.path.join(current_dir, 'GroundingDINO')
sys.path.append(project_root)

from Utils import get_video_properties
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import numpy as np
import torch
from PIL import Image
import groundingdino.datasets.transforms as T
import os
from torchvision.ops import box_convert
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))

        self.data = []
        self.targets = []

        for i, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for filename in os.listdir(class_dir):
                self.data.append(os.path.join(class_dir, filename))
                self.targets.append(i)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        target = self.targets[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, target


class DatasetGenerator:

    def __init__(self,
                 video_path: str,
                 text_prompt: str,
                 destination_folder: str,
                 ):

        self.video_path = video_path
        self.text_prompt = text_prompt
        self.destination_folder = destination_folder

    def crop(self, frame, boxes):

        h, w, _ = frame.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        min_col, min_row, max_col, max_row = map(int, xyxy[0])
        crop_image = frame[min_row:max_row, min_col:max_col, :]

        return crop_image

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

    def generate_class_dataset(self, video_path, destination_folder) -> None:

        # Load model, set up variables and get video properties
        cap, fps, width, height, fourcc = get_video_properties(video_path)
        model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                           "GroundingDINO/weights/groundingdino_swint_ogc.pth")
        TEXT_PROMPT = self.text_prompt
        BOX_TRESHOLD = 0.6
        TEXT_TRESHOLD = 0.6

        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
            print(f"Created directory: {destination_folder}")

        # Get last file number index if dataset_train folder is not empty
        files = os.listdir(destination_folder)
        import re

        if len(files) > 0:
            files.sort(key=lambda x: int(re.search(r"img_(\d+)", x).group(1)))
            file_name = files[-1]
            pattern = r"img_(\d+)"
            match = re.search(pattern, file_name)
            if match:
                number_part = match.group(1)
                index = int(number_part)
                print("Extracted number:", index)
            else:
                raise ValueError("File found without index number. The images files should be named img_index.format")

        else:
            index = 0
            print("No files found. Starting index at 0.")

        # Read video frames, crop image based on text prompt object detection and generate dataset_train
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % 60 * 3 == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_transformed = self.preprocess_image(frame_rgb)

                boxes, logits, phrases = predict(
                    model=model,
                    image=image_transformed,
                    caption=TEXT_PROMPT,
                    box_threshold=BOX_TRESHOLD,
                    text_threshold=TEXT_TRESHOLD
                )

                if boxes.size()[0] > 0:
                    crop_image = self.crop(frame, boxes)
                    cv2.imwrite(os.path.join(destination_folder, f"img_{index}.jpg"), crop_image)
                    index += 1

            frame_count += 1

    def generate_full_dataset(self):

        video_files = sorted(os.listdir(self.video_path))
        video_files = [video_file for video_file in video_files if not video_file.startswith('.')]
        for video_file in video_files:
            parts = video_file.split('_')
            class_name = parts[0]
            test = parts[1].split('.')[0] == 'test'
            if not test:
                self.generate_class_dataset(
                 os.path.join(self.video_path, video_file),
                 os.path.join(self.destination_folder, 'train', class_name)
                 )
            if test:
                self.generate_class_dataset(
                    os.path.join(self.video_path, video_file),
                    os.path.join(self.destination_folder, 'test', class_name)
                )


