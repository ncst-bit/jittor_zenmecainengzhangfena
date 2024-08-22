import os
import json
import numpy as np
import jittor as jt
import jclip as clip
from PIL import Image
import jittor.transform as transforms
from jittor.transform import CenterCrop, ImageNormalize, Compose, _setup_size, to_pil_image, resize

jt.flags.use_cuda = 1

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def to_tensor(data):
    return jt.Var(data)

class ImageToTensor(object):
    def __call__(self, input):
        input = np.asarray(input)
        if len(input.shape) < 3:
            input = np.expand_dims(input, -1)
        return to_tensor(input)

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
            size = int(new_h), int(new_w)
        return resize(img, size, self.mode)

def infer(test_imgs_dir_car,model_clip_vit_path,model_clip_rn101_path):

    model, preprocess = clip.load(model_clip_vit_path)
    model2, preprocess2 = clip.load(model_clip_rn101_path)
    model.eval()
    model2.eval()
    classes = open('./classes_b.txt').read().splitlines()

    with open("./car_cate.json",'r') as f:
        car_cate=json.load(f)

    test_transform = transforms.Compose([
        Resize(224, mode=Image.BICUBIC),
        CenterCrop(224),
        _convert_image_to_rgb,
        transforms.ImageNormalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

    new_classes=[]
    for c in classes:
        c = c.split(' ')[0]
        if c.startswith('Stanford-Cars'):
            c = c[14:]
            car_brand=c.split('_')[0]
            car_year = c.split('_')[-1]
            app_des1=", this car's Drivetrain:"+car_cate[c]['Drivetrain']
            app_des2= ", this car's Lights:"+car_cate[c]['Lights']
            if c=='Toyota_4Runner_SUV_2012':
                app_des3=",this car's Engine:V6"
                des_sum='A photo of a '+c +',a type of'+car_cate[c]['type']+ ", the brand is "+car_brand+", the model's year is"+car_year+app_des1+app_des2+app_des3+', a photo of Stanford Cars DataSet'
            elif c=='Toyota_Sequoia_SUV_2012':
                app_des3=",this car's Engine:V8"
                des_sum='A photo of a '+c +',a type of'+car_cate[c]['type']+ ", the brand is "+car_brand+", the model's year is"+car_year+app_des1+app_des2+app_des3+', a photo of Stanford Cars DataSet'
            elif c=='Ford_F-150_Regular_Cab_2007':
                app_des3=",this pickup's Engine:V6"
                des_sum='A photo of a '+c +',a type of'+car_cate[c]['type']+ ", the brand is "+car_brand+", the model's year is"+car_year+app_des1+app_des2+app_des3+', a photo of Stanford Cars DataSet'
            elif c=='Ford_Ranger_SuperCab_2011':
                app_des3=",this pickup's Engine:V8"
                des_sum='A photo of a '+c +',a type of'+car_cate[c]['type']+ ", the brand is "+car_brand+", the model's year is"+car_year+app_des1+app_des2+app_des3+', a photo of Stanford Cars DataSet'
            elif c=='Chevrolet_Corvette_Convertible_2012':
                app_des1 = ",this spider's Drivetrain:" + car_cate[c]['Drivetrain']
                des_sum='A photo of a '+c +',a type of'+car_cate[c]['type']+ ", the brand is "+car_brand+", the model's year is"+car_year+app_des1+app_des2+', a photo of Stanford Cars DataSet'
            else:
                des_sum='A photo of a '+c +',a type of'+car_cate[c]['type']+ ", the brand is "+car_brand+", the model's year is"+car_year+app_des1+app_des2+', a photo of Stanford Cars DataSet'
            new_classes.append(des_sum)

    text = clip.tokenize(new_classes)
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_features2 = model2.encode_text(text)
    text_features2 /= text_features2.norm(dim=-1, keepdim=True)

    imgs = os.listdir(test_imgs_dir_car)
    save_file = open('result_car.txt', 'w')
    for img in imgs:
        img_path = os.path.join(test_imgs_dir_car, img)
        image = Image.open(img_path)
        image = jt.Var(test_transform(image)).unsqueeze(0)
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_probs = 100.0 * jt.Var(image_features) @ text_features.transpose(0, 1)

        image_features2 = model2.encode_image(image)
        image_features2 /= image_features2.norm(dim=-1, keepdim=True)
        text_probs2 = 100.0 * jt.Var(image_features2) @ text_features2.transpose(0, 1)

        _, top_labels = (text_probs*0.8+text_probs2*1)[0].topk(5)
        save_file.write(img + ' ' +' '.join([str(p.item()) for p in top_labels]) + '\n')

    save_file.close()
