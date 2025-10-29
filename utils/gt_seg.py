from pathlib import Path
import torch
from pyvane import image, pipeline
from torchvision import tv_tensors
import torchvision.transforms.v2 as transf

class Gt_seg(pipeline.BaseProcessor):
    def __init__(self):
        transforms = transf.Compose([
            transf.PILToTensor(),   
            transf.Grayscale(num_output_channels=3),
            # transf.Resize(size=256, antialias=True),
            transf.ToDtype({tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.int64}),
            transf.Lambda(lambda x: (x > 0.5).float())
        ])

        self.transforms = transforms


    def preprocess_img(self, img):
        img = img.data
        img = tv_tensors.Image(img)
        img = self.transforms(img)
        # img = img.unsqueeze(0)
        return img

    def apply(self, img, file=None):
        img = self.preprocess_img(img)
        # scores = self.model(img)
        # pred = (torch.argmax(scores, dim=1)[0])

        new_img = image.Image(data=img[0], path=Path.cwd(), pix_size=(1.,1.))

        return new_img