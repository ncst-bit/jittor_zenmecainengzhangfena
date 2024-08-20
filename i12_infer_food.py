import os
import json
import numpy as np
import jittor as jt
import jclip as clip
from jittor import nn
from PIL import Image
from jittor.models.resnet import *
from jittor.transform import CenterCrop, _setup_size, to_pil_image, resize

jt.flags.use_cuda = 1

def _convert_image_to_rgb(image):
    return image.convert("RGB")

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
            new_w, new_h = (new_short, new_long) if w <= h else (new_long,new_short)
            size = (int(new_h), int(new_w))
        return resize(img, size, self.mode)

def get_cache_logits(model,preprocess,test_imgs_dir,adapter_f_model_path,cache_calue_path):

    adapter = nn.Linear(404, 512, bias=False)
    adapter.weight = nn.Parameter(jt.Var(jt.load(adapter_f_model_path)))
    adapter.eval()
    test_images_feature=[]
    with jt.no_grad():
        for img in os.listdir(test_imgs_dir):
            img_path = os.path.join(test_imgs_dir, img)
            image = Image.open(img_path)
            images = preprocess(image).unsqueeze(0)
            image_features = model.encode_image(images)
            for i in image_features:
                test_images_feature.append(i)

    beta,alpha=5.5,1
    test_images_feature=np.array(test_images_feature)
    test_images_feature /= jt.Var(test_images_feature).norm(dim=-1, keepdim=True)
    cache_values = np.load(cache_calue_path)
    affinity = adapter(test_images_feature)
    cache_logits = np.exp((-1) * (beta - beta * affinity)) @ cache_values

    return cache_logits


def get_clip_logits(model,preprocess,test_imgs_dir):

    with open("./food_categories.json", 'r', encoding='utf-8') as load_f:
        food_categories = json.load(load_f)
    classes = open('./classes_b.txt').read().splitlines()
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
    preds = []
    with jt.no_grad():
        for img in os.listdir(test_imgs_dir):
            img_path = os.path.join(test_imgs_dir, img)
            image = Image.open(img_path)
            image = preprocess(image).unsqueeze(0)
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_probs = 100.0 * image_features @ text_features.transpose(0, 1)
            preds.append(text_probs[0])
    return preds

def get_clip_logits2(model,preprocess,test_imgs_dir):

    with open("./food_categories.json", 'r', encoding='utf-8') as load_f:
        food_categories = json.load(load_f)
    classes = open('./classes_b.txt').read().splitlines()
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
    preds = []
    with jt.no_grad():
        for img in os.listdir(test_imgs_dir):
            img_path = os.path.join(test_imgs_dir, img)
            image = Image.open(img_path)
            image = preprocess(image).unsqueeze(0)
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_probs = 100.0 * image_features @ text_features.transpose(0, 1)
            preds.append(text_probs[0])
    return preds

def get_pesudo_label_logits(model,preprocess,test_imgs_dir):

    preds = []
    with jt.no_grad():
        for img in os.listdir(test_imgs_dir):
            img_path = os.path.join(test_imgs_dir, img)
            image = Image.open(img_path)
            image = preprocess(image).unsqueeze(0)
            output = model(image)
            preds.append(output[0])
    return preds

def infer(test_imgs_dir,model_clip_vit_path,model_clip_rn101_path,adapter_f_model_path,cache_calue_path):

    imgs = os.listdir(test_imgs_dir)

    model2, preprocess = clip.load(model_clip_rn101_path)
    model2.eval()
    clip_logits2=np.array(get_clip_logits2(model2,preprocess,test_imgs_dir))
    model2=''

    model, preprocess = clip.load(model_clip_vit_path)
    model.eval()

    classifer=Classifier(model,101)
    classifer.load_state_dict(jt.load('./model_food/food_clip.pkl'))
    classifer.eval()
    pesudo_logits=np.array(get_pesudo_label_logits(classifer,preprocess,test_imgs_dir))
    classifer=''

    cache_logits=np.array(get_cache_logits(model,preprocess,test_imgs_dir,adapter_f_model_path,cache_calue_path))
    clip_logits=np.array(get_clip_logits(model,preprocess,test_imgs_dir))

    logits = cache_logits*4+clip_logits+pesudo_logits+clip_logits2*2
    save_file = open('result_food.txt', 'w')
    for i in range(len(logits)):
        _, top_labels = jt.Var(logits[i]).topk(5)
        save_file.write(imgs[i] + ' ' + ' '.join([str(p.item()) for p in top_labels]) + '\n')
    save_file.close()

