import os
import json
import numpy as np
import jittor as jt
import jclip as clip
from jittor import nn
from PIL import Image
import jittor.transform as transforms
from jittor.transform import CenterCrop,_setup_size, to_pil_image, resize

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

def train(model,train_loader,id2class_trainset,class2id_food,num_epochs,learning_rate,classes,count):

    train_image_features=[]
    train_labels=[]
    with jt.no_grad():
        for images, labels in train_loader:
            image_features = model.encode_image(images)
            for i in image_features:
                train_image_features.append(i)
            for i in labels:
                temp=np.zeros(count)
                temp[class2id_food[id2class_trainset[int(i[0])]]]=1
                train_labels.append(temp)
    train_image_features /= jt.Var(np.array(train_image_features)).norm(dim=-1, keepdim=True)
    cache_keys = np.array(train_image_features)
    train_labels = np.array(train_labels)
    np.save("./model_food/food_cache_values.npy", train_labels)
    cache_values = jt.Var(train_labels)
    adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False)
    adapter.weight = nn.Parameter(jt.Var(cache_keys))

    with open("./food_categories.json", 'r', encoding='utf-8') as load_f:
        food_categories = json.load(load_f)

    new_classes=[]
    for c in classes:
        c = c.split(' ')[0]
        if c.startswith('Food-101'):
            c = c[9:]
            if c in food_categories.keys():
                new_classes.append('a photo of a ' + c + ', a type of ' + food_categories[c])
            else:
                new_classes.append('a photo of a ' + c + ', a type of food')

    text = clip.tokenize(new_classes)
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    clip_weights = jt.Var(np.array(text_features).T)

    loss = nn.CrossEntropyLoss()
    optimizer = jt.optim.AdamW(adapter.parameters(), lr=learning_rate, eps=1e-4)
    scheduler = jt.lr_scheduler.CosineAnnealingLR(optimizer,num_epochs*len(train_loader))
    beta,alpha=5.5,1

    for train_idx in range(num_epochs):
        adapter.train()
        print('Train Epoch: {:} / {:}'.format(train_idx, num_epochs))
        for i, (images, target) in enumerate(train_loader):
            with jt.no_grad():
                image_features = model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            affinity = adapter(image_features)
            cache_logits =  ((-1) * (beta - beta * affinity)).exp() @ cache_values
            clip_logits = 100. * image_features @ clip_weights
            tip_logits = clip_logits + cache_logits * alpha
            cur_loss = loss(tip_logits, target)
            optimizer.zero_grad()
            optimizer.backward(cur_loss)
            optimizer.step()
            scheduler.step()

    jt.save(adapter.weight,'./model_food/food_adapter_f_img.pkl')

def train_adapter_f_food(train_dir,model_clip_path):

    train_transform = transforms.Compose([
        Resize(224, mode=Image.BICUBIC),
        CenterCrop(224), _convert_image_to_rgb,
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ImageNormalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

    model_save_dir='./model_food/'
    os.makedirs(model_save_dir,exist_ok=True)

    num_epochs = 50
    learning_rate = 0.001

    train_dataset = jt.dataset.ImageFolder(root=train_dir, transform=train_transform)
    train_loader = jt.dataset.DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=4)

    class2id = train_dataset.class_to_idx
    id2class_trainset = {v: k for k, v in class2id.items()}

    model, preprocess = clip.load(model_clip_path)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    count = 0
    classes = open('./classes_b.txt').read().splitlines()
    class2id_food = {}
    for c in classes:
        c1 = c.split(' ')[0]
        if c1.startswith('Food-101'):
            c1 = c[9:].split(' ')[0]
            class2id_food [c1] = int(c.split(' ')[1]) - 143
            count+=1

    train(model,train_loader,id2class_trainset,class2id_food,num_epochs,learning_rate,classes,count)
