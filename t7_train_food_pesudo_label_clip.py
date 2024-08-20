import os
import numpy as np
import jittor as jt
import jclip as clip
from jittor import nn
from PIL import Image
from jittor.models.resnet import *
import jittor.transform as transforms
from jittor.transform import CenterCrop, ImageNormalize, Compose, _setup_size, to_pil_image, resize

jt.flags.use_cuda = 1

def _convert_image_to_rgb(image):
    return image.convert("RGB")

class Resize:

    def __init__(self, size, mode=Image.BILINEAR):
        if isinstance(size, int):
            self.size = size
        else:
            self.size = _setup_size(
                size,
                error_msg="If size is a sequence, it should have 2 values")
        self.mode = mode

    def __call__(self, img: Image.Image):
        if not isinstance(img, Image.Image):
            img = to_pil_image(img)
        if isinstance(self.size, int):
            w, h = img.size

            short, long = (w, h) if w <= h else (h, w)
            if short == self.size:
                return img

            new_short, new_long = self.size, int(self.size * long / short)
            new_w, new_h = (new_short, new_long) if w <= h else (new_long,
                                                                 new_short)
            size = (int(new_h), int(new_w))
        return resize(img, size, self.mode)

train_transform = transforms.Compose([
    Resize(224, mode=Image.BICUBIC),
    CenterCrop(224), _convert_image_to_rgb,
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ImageNormalize((0.48145466, 0.4578275, 0.40821073),(0.26862954, 0.26130258, 0.27577711))])

class Classifier(nn.Module):

    def __init__(self,clip_model,output_size):
        self.clip_model=clip_model.visual
        self.input_size=self.clip_model.output_dim
        self.fc1 = nn.Linear(self.input_size, self.input_size, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.input_size,output_size, bias=True)

    def execute(self, x):
        out = self.clip_model(x)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

def train(model_clip_path,train_dir):

    num_epochs = 50
    batch_size = 64

    model, preprocess = clip.load(model_clip_path)
    model = Classifier(model,101)

    train_dataset = jt.dataset.ImageFolder(root=train_dir, transform=train_transform)
    train_loader = jt.dataset.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    optimizer = jt.optim.AdamW(model.parameters(), lr=0.0005, eps=1e-4)
    scheduler = jt.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs * len(train_loader))
    loss = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        print('Train Epoch: {:} / {:}'.format(epoch, num_epochs))
        model.train()
        count=0
        for images, labels in train_loader:
            count+=1
            outputs = model(images)
            curl_loss = loss(outputs, labels)
            optimizer.zero_grad()
            optimizer.backward(curl_loss)
            optimizer.step()
            scheduler.step()

    model.save('./model_food/food_clip.pkl')
