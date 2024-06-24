import sys
from os.path import abspath, dirname
sys.path.append(abspath(dirname(__name__)))
import gradio as gr
import torch
from os.path import dirname
from torchvision import transforms
from PIL import Image
import yaml
from torch import nn
def image_classification():

    with open(f"{dirname(abspath(__file__))}/config.yaml", 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        labels = config["labels"]

    class DinoVisionTransformerClassifier(nn.Module):
        def __init__(self,num_classes):
            super(DinoVisionTransformerClassifier, self).__init__()
            self.transformer = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
            self.classifier = nn.Sequential(nn.Linear(384, 256), nn.ReLU(), nn.Linear(256, num_classes))

        def forward(self, x):
            x = self.transformer(x)
            x = self.transformer.norm(x)
            x = self.classifier(x)
            return x

    dino = DinoVisionTransformerClassifier(len(labels))
    model_path = f"{dirname(abspath(__file__))}/model.pth"
    state_dict = torch.load(model_path)
    dino.load_state_dict(state_dict)

    def preprocess(img_path):
        data_transforms = {
            "test": transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
                ]
            )
        }

        img = Image.open(img_path).convert('RGB')
        img_transformed = data_transforms['test'](img)

        return img_transformed

    def predict(img_path):
        img = preprocess(img_path)
        img = img.unsqueeze(0)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dino.to(device)
        dino.eval()
        with torch.no_grad():
            output = dino(img.to(device))

        _, predicted = torch.max(output.data, 1)
        print("Predicted", predicted[0])
        return labels[predicted[0].item()]

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="filepath", label="Classify Image"),
        outputs=gr.Textbox(label="Label"),
        title="Person classifier",
    )

    demo.launch(share=True, debug=True)

if __name__ == "__main__":
    image_classification()
