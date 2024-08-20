import os
import json
import shutil
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

def get_cache_logits(model,test_imgs_dir,adapter_f_model_path,cache_calue_path):

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
    cache_values = cache_values/cache_values.sum(axis=0)
    affinity = adapter(test_images_feature)
    cache_logits = np.exp((-1) * (beta - beta * affinity)) @ cache_values

    return cache_logits

def get_clip_logits(model):

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

    return text_features

def get_trainset_food_pseudo_label(model_clip_path,adapter_f_model_path,cache_calue_path):

    model, preprocess = clip.load(model_clip_path)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    text_features = get_clip_logits(model)

    count = 0
    classes = open('./classes_b.txt').read().splitlines()
    class2id_food = {}
    id2class_food = {}
    for c in classes:
        c1 = c.split(' ')[0]
        if c1.startswith('Food-101'):
            c1 = c[9:].split(' ')[0]
            class2id_food[c1] = int(c.split(' ')[1]) - 143
            id2class_food[int(c.split(' ')[1]) - 143] = c1
            count += 1

    adapter = nn.Linear(404, 512, bias=False)
    adapter.weight = nn.Parameter(jt.Var(jt.load(adapter_f_model_path)))
    adapter.eval()

    beta=5.5
    query_clip_trainset = []
    for i in open('./trainset_pseudo_labels/Food-4.txt').readlines():
        i = i.replace('\n', '')
        query_clip_trainset.append(np.load('./trainset_clip_query/' + i)[0])
    query_clip_trainset=np.array(query_clip_trainset)
    affinity = np.array(adapter(jt.Var(query_clip_trainset)))
    cache_values = np.load(cache_calue_path)
    logits1 = np.exp((-1) * (beta - beta * affinity)) @ cache_values
    logits2=100.0 * query_clip_trainset @ np.array(text_features.transpose(0, 1))
    logits=logits1+logits2
    img_list=[i.replace('\n','') for i in open('./trainset_pseudo_labels/Food-4.txt').readlines()]
    save_file = open('./trainset_pseudo_labels/Food-101.txt', 'w')
    for i in range(len(logits)):
        top_label_index = logits[i].argmax()
        label_prob = logits[i][top_label_index]
        top_label = id2class_food[top_label_index]
        save_file.write(img_list[i]+' '+str(top_label)+' '+str(label_prob) + '\n')
    save_file.close()

def get_trainset_pesudo_labels_images():

    with open("query2img.json", 'r', encoding='utf-8') as load_f:
        query2img_dict = json.load(load_f)
    f = open('./trainset_pseudo_labels/Food-101.txt').readlines()

    for i in os.listdir('./TrainSet_4_image/Food/'):
        os.makedirs('./TrainSet_4_image/Food-101-trainset-pesudo-label/'+i,exist_ok=True)
        for j in os.listdir('./TrainSet_4_image/Food/'+i):
            src='./TrainSet_4_image/Food/'+i+'/'+j
            dst='./TrainSet_4_image/Food-101-trainset-pesudo-label/'+i+'/'+j
            shutil.copy(src,dst)

    count=0
    for i in f:
        i = i.replace('\n', '')
        name = i.split(' ')[0]
        img_path=query2img_dict[name]
        predict_label = i.split(' ')[1]
        predict_label_prob = float(i.split(' ')[2])
        if predict_label_prob>33:
            shutil.copy(img_path,'./TrainSet_4_image/Food-101-trainset-pesudo-label/'+predict_label+'/'+str(count)+'_'+img_path.split('/')[-1])
            count+=1


