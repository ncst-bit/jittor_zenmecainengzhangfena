import os
import json
import shutil
import jittor as jt
import jclip as clip
from PIL import Image

jt.flags.use_cuda = 1
def get_thudog(model,preprocess,imgs_dir,output_path_thudog):

    new_classes=['a photo of a dog, a type of animal','a photo of a food','a photo of a wolf, a type of animal','a photo of a sea lion, a type of animal','a photo of a hippopotamus, a type of animal','a photo of an animal','a photo of a Caltech Dataset','a blur photo of a dalmatian','a photo of a snoppy','a photo of a catron snoppy','a blur photo of a camera','a blur photo of a chair','a photo of a car']

    text = clip.tokenize(new_classes)
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    with jt.no_grad():
        for img in os.listdir(imgs_dir):
            if img not in os.listdir(output_path_thudog):
                img_path = imgs_dir + img
                image = Image.open(img_path)
                image = preprocess(image).unsqueeze(0)
                image_features = model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_probs = (100.0 *image_features @ text_features.transpose(0, 1)).softmax(dim=-1)
                _, top_labels = text_probs[0].topk(1)
                if top_labels[0]==0:
                    shutil.copy(img_path,output_path_thudog+img)
    text_features=''

def get_food_caltech_car(classes,model,preprocess,imgs_dir,output_path_thudog,output_path_animal,output_path_food,output_path_caltech,output_path_car):

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

    for c in classes:
        c = c.split(' ')[0]
        if c.startswith('Caltech-101') and 'car' not in c:
            c = c[12:]
            if c=='minaret':
                new_classes.append('a photo of a minaret, a type of tower, a photo of Caltech-101 Dataset')
            elif c=='cellphone':
                new_classes.append('a photo of a cellphone, a type of phone, a photo of Caltech-101 Dataset')
            elif c=='inline_skate':
                new_classes.append('a photo of a inline skate , a type of shoes, a photo of Caltech-101 Dataset')
            elif c == 'binocular':
                new_classes.append('a photo of a binocular , a type of telescope, a photo of Caltech-101 Dataset')
            elif c == 'lotus':
                new_classes.append('a photo of a nelumbo , a type of flower, a photo of Caltech-101 Dataset')
            elif c == 'cougar_body':
                new_classes.append('a photo of a cougar, a type of animal, a photo of Caltech-101 Dataset')
            elif c == 'dalmatian':
                new_classes.append('a photo of a dalmatian, a type of dog')
            elif c=='wheelchair':
                new_classes.append('a photo of a wheelchair, a type of chair')
            elif c=='electric_guitar':
                new_classes.append('a photo of a guitar, a type of musical instruments, a photo of Caltech-101 Dataset')
            elif c=='grand piano':
                new_classes.append('a photo of a piano, a type of musical instruments, a photo of Caltech-101 Dataset')
            elif c=='ferry' or c=='ketch':
                new_classes.append('a photo of a ship, a photo of Caltech-101 Dataset')
            else:
                new_classes.append('a photo of a '+c+', a photo of Caltech-101 Dataset')

    new_classes.append('a photo of a airplane')
    new_classes.append('a photo of a lobster, a type of food')
    new_classes.append('a photo of a human face')
    new_classes.append('human face')
    new_classes.append('person face')
    new_classes.append('woman face')
    new_classes.append('man face')
    new_classes.append('a photo of a person face')
    new_classes.append('a photo of a woman face')
    new_classes.append('a photo of a man face')

    new_classes.append('a photo of a car')
    new_classes.append('a photo of a yellow car')
    new_classes.append('a photo of a red car')
    new_classes.append('a photo of a white car')
    new_classes.append('a photo of a dog')

    text = clip.tokenize(new_classes)
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    with jt.no_grad():
        for img in os.listdir(imgs_dir):
            if img not in os.listdir(output_path_thudog) and img not in os.listdir(output_path_animal) and img not in os.listdir(output_path_car):
                img_path = os.path.join(imgs_dir, img)
                image = Image.open(img_path)
                image = preprocess(image).unsqueeze(0)
                image_features = model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_probs = (100.0 *image_features @ text_features.transpose(0, 1)).softmax(dim=-1)
                _, top_labels = text_probs[0].topk(1)
                if top_labels[0]<101:
                    shutil.copy(imgs_dir+img,output_path_food+img)
                elif top_labels[0]==len(new_classes)-1:
                    shutil.copy(imgs_dir+img,output_path_thudog+img)
                elif top_labels[0]>len(new_classes)-5:
                    shutil.copy(imgs_dir + img, output_path_car + img)
                else:
                    shutil.copy(imgs_dir+img,output_path_caltech+img)

    text_features = ''

def get_animal(classes,model,preprocess,imgs_dir,output_path_thudog,output_path_animal,output_path_food,output_path_caltech,output_path_car):

    with open("./food_categories.json", 'r', encoding='utf-8') as load_f:
        food_categories = json.load(load_f)

    new_classes=[]
    for c in classes:
        c = c.split(' ')[0]
        if c.startswith('Animal'):
            c = c[7:]
            if c == 'Leopard':
                new_classes.append('a clear photo of a leopard, a type of animal')
            elif c == 'Chicken':
                new_classes.append('a clear photo of a hen, a type of animal')
            elif c == 'Rhinoceros':
                new_classes.append('a clear photo of a rhinoceros, a type of animal')
            elif c == 'Flamingo':
                new_classes.append('a clear photo of a flamingo, a type of bird')
            elif c == 'Shark':
                new_classes.append('a clear photo of a Shark, a type of fish')
            else:
                new_classes.append('a photo of a ' + c + ', a type of animal')

    new_classes.append('a picture of a carton shark')
    new_classes.append('a photo of a shark head')
    new_classes.append('a picture of a carton shark head')
    new_classes.append('a clear photo of two rhinoceros')
    new_classes.append('a clear photo of a bug chicken head')
    new_classes.append('a clear photo of two fish')
    new_classes.append('a clear photo of a fish')
    new_classes.append('a photo of goldfish')
    new_classes.append('a clear photo of a sea fish')
    new_classes.append('a clear photo of a fish head')
    new_classes.append('a clear photo of many colorful fish')
    new_classes.append('a clear photo of many fish')

    new_classes.append('a photo of a dog')
    new_classes.append('a blur photo of a dog')
    new_classes.append('a photo of a part of a dog')
    new_classes.append('a photo of a dog, a type of animal')
    new_classes.append('a photo of a husky, a type of dog')

    for c in classes:
        c = c.split(' ')[0]
        if c.startswith('Caltech-101') and 'car' not in c:
            c = c[12:]
            if c=='minaret':
                new_classes.append('a photo of a minaret, a type of tower, a photo of Caltech-101 Dataset')
            elif c=='cellphone':
                new_classes.append('a photo of a cellphone, a type of phone, a photo of Caltech-101 Dataset')
            elif c=='inline_skate':
                new_classes.append('a photo of a inline skate , a type of shoes, a photo of Caltech-101 Dataset')
            elif c == 'binocular':
                new_classes.append('a photo of a binocular , a type of telescope, a photo of Caltech-101 Dataset')
            elif c == 'lotus':
                new_classes.append('a photo of a nelumbo , a type of flower, a photo of Caltech-101 Dataset')
            elif c == 'cougar_body':
                new_classes.append('a photo of a cougar, a type of animal, a photo of Caltech-101 Dataset')
            elif c == 'dalmatian':
                new_classes.append('a photo of a dalmatian, a type of dog, a photo of Caltech-101 Dataset')
            elif c=='wheelchair':
                new_classes.append('a photo of a wheelchair, a type of chair, a photo of Caltech-101 Dataset')
            elif c=='electric_guitar':
                new_classes.append('a photo of a guitar, a type of musical instruments, a photo of Caltech-101 Dataset')
            elif c=='grand piano':
                new_classes.append('a photo of a piano, a type of musical instruments, a photo of Caltech-101 Dataset')
            elif c=='ferry':
                new_classes.append('a photo of a ferry, a type of ship, a photo of Caltech-101 Dataset')
            elif c=='ketch':
                new_classes.append('a photo of a ketch, a type of ship, a photo of Caltech-101 Dataset')
            elif c == 'rooster':
                new_classes.append('a picture of a rooster, a photo of Caltech-101 Dataset')
            elif c == 'rhino':
                new_classes.append('a blur photo of a rhinoceros, a type of animal, a photo of Caltech-101 Dataset')
            elif c == 'Leopards':
                new_classes.append('a blur photo of a leopards, a type of animal, a photo of Caltech-101 Dataset')
            elif c=='bass':
                new_classes.append('a blur photo of a bass, a type of fish, a photo of Caltech-101 Dataset')
            elif c=='beaver':
                new_classes.append('a blur photo of a beaver, a type of animal, a photo of Caltech-101 Dataset')
            elif c=='crocodile':
                new_classes.append('a blur photo of a crocodile, a type of crocodylus siamensis, a photo of Caltech-101 Dataset')
            elif c=='gerenuk':
                new_classes.append('a blur photo of a gerenuk, a type of animal, a photo of Caltech-101 Dataset')
            elif c == 'emu':
                new_classes.append('a blur photo of a emu, a type of animal, a photo of Caltech-101 Dataset')
            elif c=='stegosaurus':
                new_classes.append('a blur photo of a stegosaurus, a type of animal, a photo of Caltech-101 Dataset')
            elif c=='ibis':
                new_classes.append('a blur photo of a ibis, a type of bird, a photo of Caltech-101 Dataset')
            elif c=='platypus':
                new_classes.append('a blur photo of a platypus, a type of animal, a photo of Caltech-101 Dataset')
            else:
                new_classes.append('a photo of a ' + c + ', a photo of Caltech-101 Dataset')

    new_classes.append('a photo of a airplane')
    new_classes.append('a photo of a green flight airplane')
    new_classes.append('a photo of a human face')
    new_classes.append('human face')
    new_classes.append('person face')
    new_classes.append('woman face')
    new_classes.append('man face')
    new_classes.append('a photo of a person face')
    new_classes.append('a photo of a woman face')
    new_classes.append('a photo of a man face')
    new_classes.append('a blur photo of a carton platypus, a photo of Caltech-101 Dataset')

    new_classes.append('a clear photo of a food')
    new_classes.append('a photo of a baby_back_ribs, a type of food')
    new_classes.append('a photo of a hamburger, a type of food')
    new_classes.append('a photo of a chicken_curry, a type of food')
    new_classes.append('a photo of a peking_duck, a type of food')
    new_classes.append('a photo of a chicken_wings, a type of food')
    new_classes.append('a photo of a sweet, a type of food')
    new_classes.append('a photo of a cake, a type of food')
    new_classes.append('a photo of a Dessert, a type of food')
    new_classes.append('a photo of a sashimi, a type of food')
    new_classes.append('a photo of a chips, a type of food')
    new_classes.append('a photo of a fried calamari, a type of food')
    new_classes.append('a photo of a risotto, a type of food')
    new_classes.append('a photo of a mussels, a type of food')

    new_classes.append('a photo of a car')
    new_classes.append('a photo of a pickup truck')
    new_classes.append('a photo of a coupe car')
    new_classes.append('a photo of green car')
    new_classes.append('a photo of black car')
    new_classes.append('a photo of a yellow car')
    new_classes.append('a photo of a red car')
    new_classes.append('a photo of a white car')
    new_classes.append('a photo of a silver car')

    text = clip.tokenize(new_classes)
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    with jt.no_grad():
        for img in os.listdir(imgs_dir):
            if img not in os.listdir(output_path_thudog) and img not in os.listdir(output_path_caltech) and img not in os.listdir(output_path_food):
                img_path=imgs_dir+img
                image = Image.open(img_path)
                image = preprocess(image).unsqueeze(0)
                image_features = model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_probs = (100.0 *image_features @ text_features.transpose(0, 1)).softmax(dim=-1)
                _, top_labels = text_probs[0].topk(1)
                if top_labels[0]<64:
                    shutil.copy(img_path,output_path_animal+img)
                elif top_labels[0]<69:
                    shutil.copy(img_path, output_path_thudog + img)
                elif top_labels[0]>len(new_classes)-10:
                    shutil.copy(img_path, output_path_car + img)
    text_features = ''

def splitTestImageTo4DataSet(TestSetDir,model_clip_path):

    classes = open('./classes_b.txt').read().splitlines()
    model, preprocess = clip.load(model_clip_path)
    model.eval()

    output_path_thudog='./TestSetB_5_class/Thu-dog/'
    output_path_animal='./TestSetB_5_class/Animal/'
    output_path_food='./TestSetB_5_class/Food/'
    output_path_caltech='./TestSetB_5_class/Caltech/'
    output_path_car='./TestSetB_5_class/Car/'
    for path in [output_path_thudog,output_path_animal,output_path_food,output_path_caltech,output_path_car]:
        os.makedirs(path,exist_ok=True)

    get_thudog(model,preprocess,TestSetDir,output_path_thudog)
    get_animal(classes,model,preprocess,TestSetDir,output_path_thudog,output_path_animal,output_path_food,output_path_caltech,output_path_car)
    get_food_caltech_car(classes,model,preprocess,TestSetDir,output_path_thudog,output_path_animal,output_path_food,output_path_caltech,output_path_car)
    model=''