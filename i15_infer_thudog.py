import os
import numpy as np
import jittor as jt
import jclip as clip
from PIL import Image
import jittor.transform as transforms
from EfficientNetModel import EfficientNet as EfficientNet_jittor
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

def get_keys_values_clip(model_clip,keys_clip_path,values_clip_path,train_loader_clip,count,class2id_thudog,id2class_trainset):

    train_images_feature,train_labels=[],[]
    with jt.no_grad():
        for images, labels in train_loader_clip:
            image_features = model_clip.encode_image(images)
            for i in image_features:
                train_images_feature.append(i)
            for i in labels:
                temp=np.zeros(count)
                temp[class2id_thudog[id2class_trainset[int(i[0])]]]=1
                train_labels.append(temp)
    train_images_feature=np.array(train_images_feature)
    train_images_feature /= jt.Var(train_images_feature).norm(dim=-1, keepdim=True)
    np.save(keys_clip_path,np.array(train_images_feature))
    np.save(values_clip_path,np.array(train_labels))
    train_images_feature, train_labels = [], []

def get_keys_values_b6(feature_extractor_b6,keys_b6_path,values_b6_path,train_loader_b6,count,class2id_thudog,id2class_trainset):

    train_images_feature, train_labels = [], []
    with jt.no_grad():
        for images, labels in train_loader_b6:
            image_features = feature_extractor_b6(images)
            for i in image_features:
                train_images_feature.append(i.cpu())
            for i in labels:
                temp=np.zeros(count)
                temp[class2id_thudog[id2class_trainset[int(i[0])]]]=1
                train_labels.append(temp)
    train_images_feature=np.array(train_images_feature)
    train_images_feature /= jt.Var(train_images_feature).norm(dim=-1, keepdim=True)
    np.save(keys_b6_path,np.array(train_images_feature))
    np.save(values_b6_path,np.array(train_labels))
    train_images_feature, train_labels = [], []

def get_clip_weights(model_clip,clip_weights_path):

    classes = open('./classes_b.txt').read().splitlines()
    new_classes=[]
    for c in classes:
        c = c.split(' ')[0]
        if c.startswith('Thu-dog'):
            c = c[8:]
            if 'terrier' in c:
                new_classes.append('a photo a ' + c + ', a type of terrier dog')
            else:
                new_classes.append('a photo a ' + c + ', a type of dog')
    text = clip.tokenize(new_classes)
    text_features = model_clip.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    np.save(clip_weights_path,np.array(text_features))
    text_features=''

def create_trainset_pesudo_value_and_keys(trainset_pseudo_label_4_thudog_path,keys_trainset_pesudo_clip_path,keys_trainset_pesudo_b6_path,values_trainset_pesudo_clip_path,values_trainset_pesudo_b6_path,keys_clip_path,keys_b6_path,values_clip_path,values_b6_path,clip_weights_path,id2class_thudog,count,class2id_thudog):

    beta, alpha = 5.5, 1
    query_trainset_clip = []
    query_trainset_b6 = []
    for i in open(trainset_pseudo_label_4_thudog_path).readlines():
        i=i.replace('\n','')
        query_trainset_clip.append(np.load('./trainset_clip_query/'+i)[0])
        query_trainset_b6.append(np.load('./trainset_b6_query/'+i)[0])

    query_trainset_clip=np.array(query_trainset_clip)
    query_trainset_b6=np.array(query_trainset_b6)

    keys_clip=np.load(keys_clip_path)
    keys_b6=np.load(keys_b6_path)
    values_clip=np.load(values_clip_path)
    values_b6=np.load(values_b6_path )
    clip_weights=np.load(clip_weights_path)

    logits1=values_clip.T @ np.exp((-1) * (beta - beta * (keys_clip @ query_trainset_clip.T)))
    logits2=values_b6.T @ np.exp((-1) * (beta - beta * (keys_b6 @ query_trainset_b6.T)))
    logits3=100.0 * query_trainset_clip @ clip_weights.T
    logits_trainset=logits1.T * 0.5 + logits2.T * 1.5 + logits3 * 0.1

    trainset_img_list=[i.replace('\n','') for i in open('./trainset_pseudo_labels/Thu-dog-4.txt').readlines()]
    trainset_pesudo=[]
    for i in range(len(logits_trainset)):
        top_label_index = logits_trainset[i].argmax()
        label_prob = logits_trainset[i][top_label_index]
        top_label = id2class_thudog[top_label_index]
        trainset_pesudo.append([trainset_img_list[i],top_label,label_prob])

    keys_b6_trainset_pseudo = []
    values_b6_trainset_pseudo = []
    keys_clip_trainset_pseudo = []
    values_clip_trainset_pseudo = []

    for i in trainset_pesudo:
        name = i[0]
        predict_label = i[1]
        predict_label_prob = i[2]
        if float(predict_label_prob) > 8.4:
            temp = np.zeros(count)
            temp[class2id_thudog[predict_label]] = 1
            values_clip_trainset_pseudo.append(temp)
            values_b6_trainset_pseudo.append(temp)
            keys_b6_trainset_pseudo.append(np.load('./trainset_b6_query/' + name)[0])
            keys_clip_trainset_pseudo.append(np.load('./trainset_clip_query/' + name)[0])

    print(len(values_clip_trainset_pseudo))

    np.save(keys_trainset_pesudo_b6_path, np.array(keys_b6_trainset_pseudo))
    np.save(keys_trainset_pesudo_clip_path, np.array(keys_clip_trainset_pseudo))
    np.save(values_trainset_pesudo_b6_path, np.array(values_b6_trainset_pseudo))
    np.save(values_trainset_pesudo_clip_path, np.array(values_clip_trainset_pseudo))

def get_logits_adp(model_clip,feature_extractor_b6,test_img_list,train_loader_clip,train_loader_b6,count,class2id_thudog,id2class_trainset,preprocess_clip,preprocess_b6):

    model_save_dir='./model_thudog_keys/'
    query_save_dir='./model_thudog_query/'
    os.makedirs(model_save_dir,exist_ok=True)

    query_b6_crop_path = query_save_dir+'/thudog_query_crop_b6.npy'

    keys_clip_path = model_save_dir+'/thudog_keys_clip.npy'
    keys_b6_path = model_save_dir+'/thudog_keys_b6.npy'
    values_clip_path = model_save_dir+'/thudog_values_clip.npy'
    values_b6_path = model_save_dir+'/thudog_values_b6.npy'
    clip_weights_path = model_save_dir+'/thudog_clip_weight.npy'

    print('get_keys_values_clip')
    get_keys_values_clip(model_clip,keys_clip_path,values_clip_path,train_loader_clip,count,class2id_thudog,id2class_trainset)
    print('get_clip_weights')
    get_clip_weights(model_clip,clip_weights_path)
    print('get_keys_values_b6')
    get_keys_values_b6(feature_extractor_b6,keys_b6_path,values_b6_path,train_loader_b6,count,class2id_thudog,id2class_trainset)

    beta, alpha = 5.5, 1
    query_b6_crop = np.load(query_b6_crop_path)
    keys_b6 = np.load(keys_b6_path)
    values_b6 = np.load(values_b6_path)
    values_b6 = values_b6 / values_b6.sum(axis=0)
    logits_b6 = values_b6.T @ np.exp((-1) * (beta - beta * (keys_b6 @ query_b6_crop.T)))

    return logits_b6

def get_logits_trainset_pesudo(id2class_thudog,count,class2id_thudog):

    model_save_dir='./model_thudog_keys/'
    query_save_dir='./model_thudog_query/'

    os.makedirs(model_save_dir,exist_ok=True)
    os.makedirs(query_save_dir,exist_ok=True)

    query_clip_path = query_save_dir+'/thudog_query_clip.npy'
    query_b6_path = query_save_dir+'/thudog_query_b6.npy'

    query_clip_crop_path = query_save_dir+'/thudog_query_crop_clip.npy'
    query_b6_crop_path = query_save_dir+'/thudog_query_crop_b6.npy'

    keys_clip_path = model_save_dir+'/thudog_keys_clip.npy'
    keys_b6_path = model_save_dir+'/thudog_keys_b6.npy'
    values_clip_path = model_save_dir+'/thudog_values_clip.npy'
    values_b6_path = model_save_dir+'/thudog_values_b6.npy'
    clip_weights_path = model_save_dir+'/thudog_clip_weight.npy'

    trainset_pseudo_label_4_thudog_path = './trainset_pseudo_labels/Thu-dog-4.txt'
    keys_trainset_pesudo_clip_path = model_save_dir+'/thudog_keys_trainset_pesudo_clip.npy'
    keys_trainset_pesudo_b6_path = model_save_dir+'/thudog_keys_trainset_pesudo_b6.npy'
    values_trainset_pesudo_clip_path = model_save_dir+'/thudog_values_trainset_pesudo_clip.npy'
    values_trainset_pesudo_b6_path = model_save_dir+'/thudog_values_trainset_pesudo_b6.npy'
    print('create_trainset_pesudo_value_and_keys')
    create_trainset_pesudo_value_and_keys(trainset_pseudo_label_4_thudog_path, keys_trainset_pesudo_clip_path,
                                          keys_trainset_pesudo_b6_path, values_trainset_pesudo_clip_path,
                                          values_trainset_pesudo_b6_path, keys_clip_path, keys_b6_path,
                                          values_clip_path, values_b6_path, clip_weights_path, id2class_thudog, count,
                                          class2id_thudog)

    beta, alpha = 5.5, 1
    query_clip_crop = np.load(query_clip_crop_path)
    query_b6_crop = np.load(query_b6_crop_path)

    keys_b6_trainset_pseudo = np.load(keys_trainset_pesudo_b6_path)
    keys_clip_trainset_pseudo = np.load(keys_trainset_pesudo_clip_path)
    values_b6_trainset_pseudo = np.load(values_trainset_pesudo_b6_path)
    values_clip_trainset_pseudo = np.load(values_trainset_pesudo_clip_path)

    temp = values_clip_trainset_pseudo.sum(axis=0)
    temp[temp == 0] = 1
    values_clip_trainset_pseudo = values_clip_trainset_pseudo / temp
    temp = values_b6_trainset_pseudo.sum(axis=0)
    temp[temp == 0] = 1
    values_b6_trainset_pseudo = values_b6_trainset_pseudo / temp

    logits_trainset_pesudo_clip_crop = values_clip_trainset_pseudo.T @ np.exp((-1) * (beta - beta * (keys_clip_trainset_pseudo @ query_clip_crop.T)))
    logits_trainset_pesudo_b6_crop = values_b6_trainset_pseudo.T @ np.exp((-1) * (beta - beta * (keys_b6_trainset_pseudo @ query_b6_crop.T)))

    return logits_trainset_pesudo_clip_crop.T,logits_trainset_pesudo_b6_crop.T

def infer(model_clip_path,model_b6_path,train_dir,test_img_dir):

    preprocess_clip = transforms.Compose([
        Resize(224, mode=Image.BICUBIC),
        CenterCrop(224),
        _convert_image_to_rgb,
        transforms.ImageNormalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
    preprocess_b6 = transforms.Compose([
        transforms.Resize((528, 528)),
        _convert_image_to_rgb,
        transforms.ImageNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset_clip = jt.dataset.ImageFolder(root=train_dir, transform=preprocess_clip)
    train_loader_clip = jt.dataset.DataLoader(train_dataset_clip, batch_size=4, shuffle=False, num_workers=4)

    train_dataset_b6 = jt.dataset.ImageFolder(root=train_dir, transform=preprocess_b6)
    train_loader_b6 = jt.dataset.DataLoader(train_dataset_b6, batch_size=4, shuffle=False, num_workers=4)

    class_to_idx = train_dataset_clip.class_to_idx
    id2class_trainset = {v: k for k, v in class_to_idx.items()}

    classes = open('./classes_b.txt').read().splitlines()
    class2id_thudog = {}
    id2class_thudog = {}
    count = 0
    for c in classes:
        c1 = c.split(' ')[0]
        if c1.startswith('Thu-dog'):
            c1 = c[8:].split(' ')[0]
            class2id_thudog[c1] = count
            id2class_thudog[count] = c1
            count += 1

    model_clip, preprocess = clip.load(model_clip_path)
    for param in model_clip.parameters():
        param.requires_grad = False
    model_clip.eval()

    model_b6 = EfficientNet_jittor.from_name('efficientnet-b6')
    model_b6.load_state_dict(jt.load(model_b6_path))
    feature_extractor_b6 = EfficientNetFeatureExtractor(model_b6)
    model_b6.eval()
    feature_extractor_b6.eval()

    test_img_list = [i for i in os.listdir(test_img_dir)]
    logits_b6=get_logits_adp(model_clip,feature_extractor_b6,test_img_list,train_loader_clip,train_loader_b6,count,class2id_thudog,id2class_trainset,preprocess_clip,preprocess_b6)
    logits_trainset_pesudo_clip,logits_trainset_pesudo_b6=get_logits_trainset_pesudo(id2class_thudog,count,class2id_thudog)
    logits = logits_b6.T*2+logits_trainset_pesudo_clip+logits_trainset_pesudo_b6

    save_file = open('result_thudog.txt', 'w')
    preds = []
    for i in range(len(logits)):
        top_labels = logits[i].argsort()[-5:][::-1]
        preds.append(top_labels)
        save_file.write(test_img_list[i] + ' ' + ' '.join([str(p.item()) for p in top_labels]) + '\n')
    save_file.close()