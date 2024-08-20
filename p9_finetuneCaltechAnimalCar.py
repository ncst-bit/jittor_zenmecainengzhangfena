import os
import numpy as np
import jittor as jt
import jclip as clip
from jittor import nn
from PIL import Image
from jittor.models.resnet import *
import jittor.transform as transforms

jt.flags.use_cuda = 1

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def get_cache_logits(model,test_imgs_dir,adapter_f_model_path,cache_calue_path):

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        _convert_image_to_rgb,
        transforms.ImageNormalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

    adapter = nn.Linear(352, 512, bias=False)
    adapter.weight = nn.Parameter(jt.Var(jt.load(adapter_f_model_path)))
    adapter.eval()
    test_images_feature = []
    with jt.no_grad():
        for img in os.listdir(test_imgs_dir):
            img_path = os.path.join(test_imgs_dir, img)
            image = Image.open(img_path)
            images = jt.Var(test_transform(image)).unsqueeze(0)
            image_features = model.encode_image(images)
            for i in image_features:
                test_images_feature.append(i)
    beta, alpha = 5.5, 1
    test_images_feature = np.array(test_images_feature)
    test_images_feature /= jt.Var(test_images_feature).norm(dim=-1, keepdim=True)
    cache_values = np.load(cache_calue_path)
    affinity = adapter(test_images_feature)
    cache_logits = np.exp((-1) * (beta - beta * affinity)) @ cache_values
    return cache_logits

def get_clip_logits(model,test_imgs_dir):

    classes = open('./classes_b.txt').read().splitlines()
    class2id_caltech = {}
    count = 0
    for c in classes:
        c1 = c.split(' ')[0]
        if c1.startswith('Caltech-101'):
            c1 = c[12:].split(' ')[0]
            class2id_caltech[c1] = count
            count += 1

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        _convert_image_to_rgb,
        transforms.ImageNormalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

    new_classes = []
    for c in classes:
        c = c.split(' ')[0]
        if c.startswith('Caltech-101'):
            c = c[12:]
            if c == 'minaret':
                new_classes.append('a photo of a minaret, a type of tower, a photo of Caltech-101 Dataset')
            elif c == 'cellphone':
                new_classes.append(
                    'a photo of a cellphone, a type of push-button phone, a photo of Caltech-101 Dataset')
            elif c == 'inline_skate':
                new_classes.append(
                    'a photo of a inline skate , a type of Roller skates, a photo of Caltech-101 Dataset')
            elif c == 'sea_horse':
                new_classes.append('a photo of a hippocampus , a type of animal, a photo of Caltech-101 Dataset')
            elif c == 'binocular':
                new_classes.append('a photo of a binocular , a type of telescope, a photo of Caltech-101 Dataset')
            else:
                new_classes.append('a photo of a ' + c + ', a photo of Caltech-101 Dataset')

    text = clip.tokenize(new_classes)
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    preds = []
    with jt.no_grad():
        for img in os.listdir(test_imgs_dir,):
            img_path = os.path.join(test_imgs_dir, img)
            image = Image.open(img_path)
            image = jt.Var(test_transform(image)).unsqueeze(0)
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_probs = (100.0 * image_features @ text_features.transpose(0, 1))#.softmax(dim=-1)
            preds.append(text_probs[0])
    text_features = ''
    return preds

def get_res50_logits(train_dir,model_res_50_path,test_imgs_dir,calss2id_caltech):

    def _preprocess_image(image):
        width, height = image.size
        max_size = max(width, height)
        new_image = Image.new('RGB', (max_size, max_size), (0, 0, 0))
        new_image.paste(image, ((max_size - width) // 2, (max_size - height) // 2))
        new_image = new_image.resize((224, 224))
        return new_image

    test_transform = transforms.Compose([
        _preprocess_image,
        transforms.ImageNormalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

    train_dataset = jt.dataset.ImageFolder(root=train_dir, transform=test_transform)
    class2idx = train_dataset.class_to_idx
    id2class_trainset = {v: k for k, v in class2idx.items()}

    model_res50 = Resnet50(pretrained=False)
    num_features = model_res50.fc.in_features
    model_res50.fc = nn.Linear(num_features, 91)
    model_res50.load_state_dict(jt.load(model_res_50_path))
    model_res50.eval()

    preds = []
    with jt.no_grad():
        for img in os.listdir(test_imgs_dir):
            img_path = os.path.join(test_imgs_dir, img)
            image = Image.open(img_path)
            image = jt.Var(np.array([test_transform(image)]))
            outputs = model_res50(image)
            arr = np.array(outputs.data[0])
            temp = {}
            for count, i in enumerate(arr):
                temp[id2class_trainset[count]] = i
            normalized_arr = [temp[i] for i in calss2id_caltech.keys()]
            preds.append(normalized_arr)
    model_res50=''

    return preds

def get_Animal_Caltech(test_imgs_dir,train_dir,model_res_50_path,test_imgs_dir_caltech):

    classes = open('./classes_b.txt').read().splitlines()
    class2id_caltech = {}
    count = 0
    for c in classes:
        c1 = c.split(' ')[0]
        if c1.startswith('Caltech-101'):
            c1 = c[12:].split(' ')[0]
            class2id_caltech[c1] = count
            count += 1

    model, preprocess = clip.load("./publicModel/ViT-B-32.pkl")
    model.eval()

    imgs = os.listdir(test_imgs_dir)
    res50_logits = np.array(get_res50_logits(train_dir,model_res_50_path,test_imgs_dir,class2id_caltech))

    adapter_f_model_path='./model_caltech/caltech_adapter_f_img.pkl'
    cache_calue_path='./model_caltech/caltech_cache_values.npy'
    cache_logits = np.array(get_cache_logits(model,test_imgs_dir,adapter_f_model_path,cache_calue_path))
    clip_logits = np.array(get_clip_logits(model,test_imgs_dir))
    model = ''

    logits = cache_logits+clip_logits+res50_logits
    for i in range(len(logits)):
        top_labels = np.array(logits[i])
        top_labels.sort()
        if top_labels[::-1][0]>39:
            os.rename(test_imgs_dir+imgs[i],test_imgs_dir_caltech+imgs[i])

def get_Car_Caltech(test_imgs_dir,train_dir,model_res_50_path,test_imgs_dir_caltech):

    classes = open('./classes_b.txt').read().splitlines()
    class2id_caltech = {}
    count = 0
    for c in classes:
        c1 = c.split(' ')[0]
        if c1.startswith('Caltech-101'):
            c1 = c[12:].split(' ')[0]
            class2id_caltech[c1] = count
            count += 1

    model, preprocess = clip.load("./publicModel/ViT-B-32.pkl")
    model.eval()

    imgs = os.listdir(test_imgs_dir)
    res50_logits = np.array(get_res50_logits(train_dir,model_res_50_path,test_imgs_dir,class2id_caltech))

    adapter_f_model_path='./model_caltech/caltech_adapter_f_img.pkl'
    cache_calue_path='./model_caltech/caltech_cache_values.npy'
    cache_logits = np.array(get_cache_logits(model,test_imgs_dir,adapter_f_model_path,cache_calue_path))
    clip_logits = np.array(get_clip_logits(model,test_imgs_dir))
    model = ''

    logits = cache_logits+clip_logits+res50_logits
    for i in range(len(logits)):
        top_labels = np.array(logits[i])
        top_labels.sort()
        if top_labels[::-1][0]>40:
            os.rename(test_imgs_dir+imgs[i],test_imgs_dir_caltech+imgs[i])

def finetune(test_imgs_dir_car,test_imgs_dir_animal,train_dir,model_res_50_path,test_imgs_dir_caltech):

    get_Animal_Caltech(test_imgs_dir_animal,train_dir,model_res_50_path,test_imgs_dir_caltech)
    get_Car_Caltech(test_imgs_dir_car,train_dir,model_res_50_path,test_imgs_dir_caltech)



