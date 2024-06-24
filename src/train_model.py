import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision import transforms
import sys
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')
class DinoVisionTransformerClassifier(nn.Module):
    def __init__(self,num_classes):
        super(DinoVisionTransformerClassifier, self).__init__()
        self.transformer = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.classifier = nn.Sequential(nn.Linear(384, 256), nn.ReLU(), nn.Linear(256, num_classes)))

    def forward(self, x):
        x = self.transformer(x)
        x = self.transformer.norm(x)
        x = self.classifier(x)
        return x

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

        self.data = self.data * 2
        self.targets = self.targets * 2
        self.flip_flag = [0] * (len(self.data) // 2) + [1] * (len(self.data) // 2)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        target = self.targets[idx]
        img = Image.open(img_path).convert('RGB')

        if self.flip_flag[idx]:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        if self.transform:
            img = self.transform(img)

        return img, target

class ModelTrainer:

    def __init__(self):
        pass

    def train_model(self, train_dataset):


        # Define train-test split
        #train_size = int(0.8 * len(train_dataset))
        #test_size = len(train_dataset) - train_size
        #train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size]

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

        test_dataset = self.return_torch_dataset_test()
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

        class_names = train_dataset.classes

        print(class_names)

        model = DinoVisionTransformerClassifier(len(class_names))

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device: ", device)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-6)

        num_epochs = 1

        epoch_losses = []
        for epoch in range(num_epochs):
            print("Epoch: ", epoch)
            running_loss = 0.0
            batch_losses = []
            for i, data in enumerate(train_loader):

                # get the input batch and the labels
                batch_of_images, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # model prediction
                output = model(batch_of_images.to(device))

                # compute loss and do gradient descent
                loss = criterion(output, labels.to(device))
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 50 == 49:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 50:.3f}')
                    running_loss = 0.0

        correct = 0
        total = 0

        # Limiting to the first 20 batches
        for i, data in enumerate(test_loader):
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images.to(device))
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.to("cpu") == labels).sum().item()

        print(f'Accuracy of the network on the {i * test_loader.batch_size} images: {100 * correct // total} %')

        return model

    def return_torch_dataset(self) -> CustomImageDataset:

        data_transforms = {
            "train": transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
                ]
            )
        }

        root_dir = "dataset/train"
        train_dataset = CustomImageDataset(root_dir, transform=data_transforms["train"])

        return train_dataset

    def return_torch_dataset_test(self) -> CustomImageDataset:

        data_transforms = {
            "test": transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
                ]
            )
        }

        root_dir = "dataset/test"
        train_dataset = CustomImageDataset(root_dir, transform=data_transforms["test"])

        return train_dataset

    def evaluate_model(self, model):

        test_dataset = self.return_torch_dataset_test()
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        correct = 0
        total = 0
        model = model.to(device)
        # Limiting to the first 20 batches
        for i, data in enumerate(test_loader):
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images.to(device))
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.to("cpu") == labels).sum().item()

        print(f'Accuracy of the network on the {i * test_loader.batch_size} images: {100 * correct // total} %')

        return 100 * correct // total
