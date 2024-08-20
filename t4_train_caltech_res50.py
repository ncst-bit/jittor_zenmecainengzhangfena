import os
import jittor as jt
from jittor import nn
from PIL import Image
from jittor.models.resnet import *
import jittor.transform as transforms

jt.flags.use_cuda = 1

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _preprocess_image(image):
    width, height = image.size
    max_size = max(width, height)
    new_image = Image.new('RGB', (max_size, max_size), (0, 0, 0))
    new_image.paste(image, ((max_size - width) // 2, (max_size - height) // 2))
    new_image = new_image.resize((224, 224))
    return new_image

def train(model,train_loader,num_epochs,learning_rate):

    loss = nn.CrossEntropyLoss()
    optimizer = jt.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        print('Train Epoch: {:} / {:}'.format(epoch, num_epochs))
        model.train()
        for images, labels in train_loader:
            outputs = model(images)
            curl_loss = loss(outputs, labels)
            optimizer.zero_grad()
            optimizer.backward(curl_loss)
            optimizer.step()
    model.save('./model_caltech/caltech_res50.pkl')

def train_caltch_res50(train_dir):

    model_save_dir='./model_caltech/'
    os.makedirs(model_save_dir,exist_ok=True)

    model = Resnet50(pretrained=False)
    model.load('./publicModel/resnet50.pkl')
    #https://cg.cs.tsinghua.edu.cn/jittor/assets/build/checkpoints/resnet50.pkl

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 91)
    num_epochs = 20
    learning_rate = 0.00005

    train_transform = transforms.Compose([
        _preprocess_image,
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ImageNormalize((0.48145466, 0.4578275, 0.40821073),(0.26862954, 0.26130258, 0.27577711))])
    train_dataset = jt.dataset.ImageFolder(root=train_dir, transform=train_transform)
    train_loader = jt.dataset.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    train(model,train_loader,num_epochs,learning_rate)
