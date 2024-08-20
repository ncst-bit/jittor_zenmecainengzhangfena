import os
import numpy as np
import jittor as jt
import jclip as clip
from PIL import Image
import jittor.transform as transforms
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
            new_w, new_h = (new_short, new_long) if w <= h else (new_long,new_short)
            size = int(new_h*1.2), int(new_w*1.2)
        return resize(img, size, self.mode)

def get_cache_logits(train_loader,imgs_dir,class2id_animal,idx2class_trainset,model,count):

    test_transform = transforms.Compose([
        Resize(224, mode=Image.BICUBIC),
        CenterCrop(224), _convert_image_to_rgb,
        transforms.ImageNormalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

    train_images_feature=[]
    train_labels=[]
    with jt.no_grad():
        for images, labels in train_loader:
            image_features = model.encode_image(images)
            for i in image_features:
                train_images_feature.append(i)
            for i in labels:
                temp=np.zeros(count+6)
                temp[class2id_animal[idx2class_trainset[int(i[0])]]]=1
                train_labels.append(temp)
    train_labels=np.array(train_labels)

    test_images_feature=[]
    test_labels=[]
    with jt.no_grad():
        for img in os.listdir(imgs_dir):
            img_path = os.path.join(imgs_dir, img)
            image = Image.open(img_path)
            images = jt.Var(test_transform(image)).unsqueeze(0)
            image_features = model.encode_image(images)
            for i in image_features:
                test_images_feature.append(i)
            for i in labels:
                temp = np.zeros(count)
                temp[class2id_animal[idx2class_trainset[int(i[0])]]]=1
                test_labels.append(temp)

    beta,alpha=5.5,1
    train_images_feature=np.array(train_images_feature)
    train_images_feature /= jt.Var(train_images_feature).norm(dim=-1, keepdim=True)
    test_images_feature=np.array(test_images_feature)
    test_images_feature /= jt.Var(test_images_feature).norm(dim=-1, keepdim=True)
    affinity = np.array(train_images_feature.data) @ np.array(test_images_feature.data).T
    cache_logits = train_labels.T @ np.exp((-1) * (beta - beta * affinity))

    return cache_logits

def get_clip_logits(imgs_dir,model):

    test_transform = transforms.Compose([
        Resize(224, mode=Image.BICUBIC),
        CenterCrop(224), _convert_image_to_rgb,
        transforms.ImageNormalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

    classes = open('./classes_b.txt').read().splitlines()
    new_classes=[]
    for c in classes:
        c = c.split(' ')[0]
        if c.startswith('Animal'):
            c = c[7:]
            if c == 'Leopard':
                new_classes.append('a clear photo of a ' + c + ', is not a cougar ,a type of animal')
            elif c == 'Elk':
                new_classes.append('a photo of a' + c + ', is not a Caribou, a type of deer')
            elif c == 'Caribou':
                new_classes.append('a photo of a' + c + ', is not a Elk, a type of deer')
            elif c == 'Sea_lion':
                new_classes.append('a photo of a' + c + ', is not a Seal, a type of animal')
            elif c == 'Fox':
                new_classes.append('a clear photo of a' + c + ', a type of canidae animal')
            elif c == 'Shark':
                new_classes.append('a clear photo of a carton Shark, a type of fish')
            elif c == 'Wolf':
                new_classes.append('a clear photo of a' + c + ', a type of canidae animal')
            else:
                new_classes.append('a photo of a ' + c + ', a type of animal')
    new_classes.append('a photo of a shark, a type of fish')
    new_classes.append('a picture of a carton shark head')
    new_classes.append('a photo of a stone eagle, a type of animal')
    new_classes.append('a photo of a carton whale, a type of fish')
    new_classes.append("a photo of dolphin's dorsal fin, a type of fish")
    new_classes.append("a photo of dolphin's tail fin, a type of fish")

    text = clip.tokenize(new_classes)
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    preds = []
    with jt.no_grad():
        for img in os.listdir(imgs_dir):
            img_path = os.path.join(imgs_dir, img)
            image = Image.open(img_path)
            image = jt.Var(test_transform(image)).unsqueeze(0)
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_probs = (100.0 * image_features @ text_features.transpose(0, 1))
            preds.append(text_probs[0])

    return preds

def infer(train_dir,test_dir_animal,model_clip_path):

    test_transform = transforms.Compose([
        Resize(224, mode=Image.BICUBIC),
        CenterCrop(224), _convert_image_to_rgb,
        transforms.ImageNormalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

    train_dataset = jt.dataset.ImageFolder(root=train_dir, transform=test_transform)
    train_loader = jt.dataset.DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4)

    classes = open('./classes_b.txt').read().splitlines()
    class2id_animal={}
    id2class_animal={}
    count = 0
    for c in classes:
        c1 = c.split(' ')[0]
        if c1.startswith('Animal'):
            c1 = c[7:].split(' ')[0]
            class2id_animal[c1] = count
            id2class_animal[count] = c1
            count += 1

    class_to_idx = train_dataset.class_to_idx
    idx2class_trainset = {v: k for k, v in class_to_idx.items()}

    model, preprocess = clip.load(model_clip_path)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    imgs=os.listdir(test_dir_animal)
    cache_logits=np.array(get_cache_logits(train_loader,test_dir_animal,class2id_animal,idx2class_trainset,model,count))
    clip_logits=np.array(get_clip_logits(test_dir_animal,model))

    save_file = open('result_animal.txt', 'w')
    logits=cache_logits.T+clip_logits
    for i in range(len(logits)):
        _, top_labels = jt.Var(logits[i]).topk(5)
        save_file.write(imgs[i] + ' ' + ' '.join([str(p.item()) for p in top_labels]) + '\n')
    save_file.close()
