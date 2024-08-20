import os
import numpy as np
import jittor as jt
import jclip as clip
from jittor import nn
import jittor.transform as transforms

jt.flags.use_cuda = 1

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def train(model,train_loader,id2class_trainset,class2id_caltech,num_epochs,learning_rate,classes,count):

    train_image_features=[]
    train_labels=[]
    with jt.no_grad():
        for images, labels in train_loader:
            image_features = model.encode_image(images)
            for i in image_features:
                train_image_features.append(i)
            for i in labels:
                temp=np.zeros(count)
                temp[class2id_caltech[id2class_trainset[int(i[0])]]]=1
                train_labels.append(temp)
    train_image_features /= jt.Var(np.array(train_image_features)).norm(dim=-1, keepdim=True)
    cache_keys = np.array(train_image_features)
    train_labels = np.array(train_labels)
    np.save("./model_caltech/caltech_cache_values.npy", train_labels)
    cache_values = jt.Var(np.array(train_labels))

    adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False)
    adapter.weight = nn.Parameter(jt.Var(cache_keys))

    new_classes=[]
    for c in classes:
        c = c.split(' ')[0]
        if c.startswith('Caltech-101'):
            c = c[12:]
            if c=='minaret':
                new_classes.append('a photo of a minaret, a type of tower, a photo of Caltech-101 Dataset')
            elif c=='cellphone':
                new_classes.append('a photo of a cellphone, a type of push-button phone, a photo of Caltech-101 Dataset')
            elif c=='inline_skate':
                new_classes.append('a photo of a inline skate , a type of Roller skates, a photo of Caltech-101 Dataset')
            elif c == 'sea_horse':
                new_classes.append('a photo of a hippocampus , a type of animal, a photo of Caltech-101 Dataset')
            elif c == 'binocular':
                new_classes.append('a photo of a binocular , a type of telescope, a photo of Caltech-101 Dataset')
            else:
                new_classes.append('a photo of a '+c+', a photo of Caltech-101 Dataset')

    text = clip.tokenize(new_classes)
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    clip_weights = jt.Var(np.array(text_features).T)

    optimizer = jt.optim.AdamW(adapter.parameters(), lr=learning_rate, eps=1e-4)
    scheduler = jt.lr_scheduler.CosineAnnealingLR(optimizer,num_epochs*len(train_loader))
    loss = nn.CrossEntropyLoss()
    beta,alpha=5.5,1

    for train_idx in range(num_epochs):
        adapter.train()
        print('Train Epoch: {:} / {:}'.format(train_idx, 50))
        for i, (images, target) in enumerate(train_loader):
            with jt.no_grad():
                image_features = model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            affinity = adapter(image_features)
            cache_logits =  ((-1) * (beta - beta * affinity)).exp() @ cache_values
            clip_logits = 100. * image_features @ clip_weights
            tip_logits = clip_logits + cache_logits * alpha

            target = jt.Var(np.array([class2id_caltech[id2class_trainset[int(i)]] for i in target.data]))
            cur_loss = loss(tip_logits, target)
            optimizer.zero_grad()
            optimizer.backward(cur_loss)
            optimizer.step()
            scheduler.step()

    jt.save(adapter.weight,'./model_caltech/caltech_adapter_f_img.pkl')

def train_adapter_f_caltech(train_dir):

    num_epochs = 50
    learning_rate = 0.001
    model_save_dir='./model_caltech/'
    os.makedirs(model_save_dir,exist_ok=True)

    train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        _convert_image_to_rgb,
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ImageNormalize((0.48145466, 0.4578275, 0.40821073),(0.26862954, 0.26130258, 0.27577711))])

    train_dataset = jt.dataset.ImageFolder(root=train_dir, transform=train_transform)
    train_loader = jt.dataset.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    class2id_trainset = train_dataset.class_to_idx
    id2class_trainset = {v: k for k, v in class2id_trainset.items()}

    model, preprocess = clip.load("./publicModel/ViT-B-32.pkl")
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    classes = open('./classes_b.txt').read().splitlines()
    class2id_caltech={}
    count = 0
    for c in classes:
        c1 = c.split(' ')[0]
        if c1.startswith('Caltech-101'):
            c1 = c[12:].split(' ')[0]
            class2id_caltech[c1] = count
            count += 1

    train(model,train_loader,id2class_trainset,class2id_caltech,num_epochs,learning_rate,classes,count)
