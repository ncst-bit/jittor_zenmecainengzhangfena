import os
import json
import numpy as np
import jittor as jt
import jclip as clip
from PIL import Image
import jittor.transform as transforms
from EfficientNetModel import EfficientNet as EfficientNet_jittor
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

class EfficientNetFeatureExtractor(jt.nn.Module):
    def __init__(self, original_model):
        super(EfficientNetFeatureExtractor, self).__init__()
        self.stem = original_model._conv_stem
        self.bn0 = original_model._bn0
        self.blocks = original_model._blocks
        self.head_conv = original_model._conv_head
        self.head_bn = original_model._bn1
        self._avg_pooling = original_model._avg_pooling
    def execute(self, x):
        x = self.stem(x)
        x = self.bn0(x)
        for block in self.blocks:
            x = block(x)
        x = self.head_conv(x)
        x = self.head_bn(x)
        x = self._avg_pooling(x)
        x = x.flatten(start_dim=1)
        return x

def get_query_b6(feature_extractor_b6,output_path,imgs):

    preprocess_b6 = transforms.Compose([
        transforms.Resize((528, 528)),
        _convert_image_to_rgb,
        transforms.ImageNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with jt.no_grad():
        for img_path in imgs:
            output_file_path=img_path.split('/')[-3]+'-'+img_path.split('/')[-2]+'-'+img_path.split('/')[-1].split('.')[0]+'.npy'
            test_images_feature = []
            image = Image.open(img_path)
            images = jt.Var(preprocess_b6(image)).unsqueeze(0)
            image_features = feature_extractor_b6(images)
            for i in image_features:
                test_images_feature.append(i.cpu())
            test_images_feature = np.array(test_images_feature)
            test_images_feature /= jt.Var(test_images_feature).norm(dim=-1, keepdim=True)
            np.save(output_path+'/'+output_file_path,test_images_feature)

def get_query_clip(model_clip,output_path,imgs):

    preprocess_clip = transforms.Compose([
        Resize(224, mode=Image.BICUBIC),
        CenterCrop(224),
        _convert_image_to_rgb,
        transforms.ImageNormalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

    with jt.no_grad():
        for img_path in imgs:
            output_file_path=img_path.split('/')[-3]+'-'+img_path.split('/')[-2]+'-'+img_path.split('/')[-1].split('.')[0]+'.npy'
            test_images_feature = []
            image = Image.open(img_path)
            images = jt.Var(preprocess_clip(image)).unsqueeze(0)
            image_features = model_clip.encode_image(images)
            for i in image_features:
                test_images_feature.append(i.cpu())
            test_images_feature = np.array(test_images_feature)
            test_images_feature /= jt.Var(test_images_feature).norm(dim=-1, keepdim=True)
            np.save(output_path+'/'+output_file_path,test_images_feature)

def get_query2img_dict(train_dir):

    imgs=[]
    query2img_dict={}
    for dirpath, _, filenames in os.walk(train_dir):
        for filename in filenames:
            if filename.endswith(('jpg', 'jpeg', 'png', 'bmp', 'tiff')):
                img_path=os.path.join(dirpath, filename)
                imgs.append(img_path)
                query_path=img_path.split('/')[-3]+'-'+img_path.split('/')[-2]+'-'+img_path.split('/')[-1].split('.')[0]+'.npy'
                query2img_dict[query_path]=img_path

    with open("query2img.json",'w',encoding='utf-8') as f:
        json.dump(query2img_dict, f,ensure_ascii=False)
    return imgs

def get_trainset_clip_query(imgs,output_path_clip,model_clip_path):

    #clip模型提取的训练集特征路径
    os.makedirs(output_path_clip,exist_ok=True)
    model_clip, preprocess = clip.load(model_clip_path)
    for param in model_clip.parameters():
        param.requires_grad = False
    model_clip.eval()
    get_query_clip(model_clip,output_path_clip,imgs)
    model_clip=''

def get_trainset_b6_query(imgs,output_path_b6,model_b6_path):

    #efficinet-b6模型提取的训练集特征路径
    os.makedirs(output_path_b6,exist_ok=True)
    model_b6 = EfficientNet_jittor.from_name('efficientnet-b6')
    model_b6.load_state_dict(jt.load(model_b6_path))
    feature_extractor_b6 = EfficientNetFeatureExtractor(model_b6)
    feature_extractor_b6.eval()
    get_query_b6(feature_extractor_b6,output_path_b6,imgs)
    model_b6=''
    feature_extractor_b6=''