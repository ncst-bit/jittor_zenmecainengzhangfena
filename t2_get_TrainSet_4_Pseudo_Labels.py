import os
import json
import numpy as np
import jittor as jt
import jclip as clip

jt.flags.use_cuda = 1
def get_thudog(trainset_query_dir,output_path_thudog,model_clip_path):

    model, preprocess = clip.load(model_clip_path)
    new_classes = ['a photo of a dog, a type of an animal', 'a photo of a food', 'a photo of an animal',
                   'a photo of a Caltech Dataset', 'a blur photo of a dalmatian', 'a photo of a catron snoppy',
                   'a blur photo of a camera', 'a blur photo of a chair', 'a photo of a car']
    text = clip.tokenize(new_classes)
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    image_features=[]
    file_list= os.listdir(trainset_query_dir)
    for file in file_list:
        image_features.append(np.load(trainset_query_dir+'/'+file)[0])
    image_features=np.array(image_features)
    text_features=np.array(text_features.transpose(0, 1).data)
    text_probs = 100.0 *(image_features @ text_features)

    thu_dog_filelist=[]
    for count,prob in enumerate(text_probs):
        if prob.argmax()==0:
            thu_dog_filelist.append(file_list[count])

    f=open(output_path_thudog,'w')
    f.write('\n'.join(thu_dog_filelist))
    f.close()
    model=''

def get_animal(trainset_query_dir,output_path_thudog,output_path_animal,model_clip_path):

    model, preprocess = clip.load(model_clip_path)
    classes = open('./classes_b.txt').read().splitlines()
    new_classes = []
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
    new_classes.append('a photo of a word:shark, a type of fish')
    new_classes.append('a photo of a stone eagle, a type of animal')
    new_classes.append('a photo of a carton whale, a type of fish')
    new_classes.append("a photo of dolphin's dorsal fin, a type of fish")
    new_classes.append("a photo of dolphin's tail fin, a type of fish")

    for c in classes:
        c = c.split(' ')[0]
        if c.startswith('Caltech-101'):
            c = c[12:]
            if c not in ['Leopards', 'bass', 'beaver', 'crayfish', 'sea_horse', 'wild_cat', 'cougar_body',
                         'cougar_face', 'ibis', 'rooster', 'rhino', 'okapi', 'platypus', 'lobster', 'gerenuk']:
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

    new_classes.append('a blur photo of a food')
    new_classes.append('a photo of a baby_back_ribs, a type of food')
    new_classes.append('a photo of a hamburger, a type of food')
    new_classes.append('a photo of a chicken_curry, a type of food')
    new_classes.append('a photo of a peking_duck, a type of food')
    new_classes.append('a photo of a chicken_wings, a type of food')
    new_classes.append('a photo of a sweet, a type of food')
    new_classes.append('a photo of a cake, a type of food')
    new_classes.append('a photo of a Dessert, a type of food')
    new_classes.append('a photo of a car')

    text = clip.tokenize(new_classes)
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    image_features,file_list=[],[]
    thudog_file_list=[i.replace('\n','') for i in open(output_path_thudog,'r').readlines()]
    for file in os.listdir(trainset_query_dir):
        if file not in thudog_file_list:
            image_features.append(np.load(trainset_query_dir+'/'+file)[0])
            file_list.append(file)
    image_features=np.array(image_features)
    text_features=np.array(text_features.transpose(0, 1).data)
    text_probs = 100.0 *(image_features @ text_features)

    animal_filelist=[]
    for count,prob in enumerate(text_probs):
        if prob.argmax()<58:
            animal_filelist.append(file_list[count])

    f=open(output_path_animal,'w')
    f.write('\n'.join(animal_filelist))
    f.close()

def get_food_caltech(trainset_query_dir,output_path_thudog,output_path_animal,output_path_food,output_path_caltech,model_clip_path):

    model, preprocess = clip.load(model_clip_path)
    classes = open('./classes_b.txt').read().splitlines()
    with open("./food_categories.json", 'r', encoding='utf-8') as load_f:
        food_categories = json.load(load_f)

    new_classes = []
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
            if c == 'minaret':
                new_classes.append('a photo of a minaret, a type of tower, a photo of Caltech-101 Dataset')
            elif c == 'cellphone':
                new_classes.append('a photo of a cellphone, a type of phone, a photo of Caltech-101 Dataset')
            elif c == 'inline_skate':
                new_classes.append('a photo of a inline skate , a type of shoes, a photo of Caltech-101 Dataset')
            elif c == 'binocular':
                new_classes.append('a photo of a binocular , a type of telescope, a photo of Caltech-101 Dataset')
            elif c == 'lotus':
                new_classes.append('a photo of a nelumbo , a type of flower, a photo of Caltech-101 Dataset')
            elif c == 'cougar_body':
                new_classes.append('a photo of a cougar, a type of animal, a photo of Caltech-101 Dataset')
            elif c == 'dalmatian':
                new_classes.append('a photo of a dalmatian, a type of dog')
            elif c == 'wheelchair':
                new_classes.append('a photo of a wheelchair, a type of chair')
            elif c == 'Faces' or c == 'Faces_easy':
                new_classes.append('a photo of a human face, a photo of Caltech-101 Dataset')
            elif c == 'electric_guitar':
                new_classes.append('a photo of a guitar, a type of musical instruments, a photo of Caltech-101 Dataset')
            elif c == 'grand piano':
                new_classes.append('a photo of a piano, a type of musical instruments, a photo of Caltech-101 Dataset')
            elif c == 'ferry' or c == 'ketch':
                new_classes.append('a photo of a ship, a photo of Caltech-101 Dataset')
            else:
                new_classes.append('a photo of a ' + c + ', a photo of Caltech-101 Dataset')
    new_classes.append('a blur photo of a carton scorpion')

    text = clip.tokenize(new_classes)
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    image_features,file_list=[],[]
    thudog_file_list=[i.replace('\n','') for i in open(output_path_thudog,'r').readlines()]
    animal_file_list=[i.replace('\n','') for i in open(output_path_animal,'r').readlines()]
    for file in os.listdir(trainset_query_dir):
        if file not in thudog_file_list and file not in animal_file_list:
            image_features.append(np.load(trainset_query_dir+'/'+file)[0])
            file_list.append(file)
    image_features=np.array(image_features)
    text_features=np.array(text_features.transpose(0, 1).data)
    text_probs = 100.0 *(image_features @ text_features)

    food_filelist,caltech_filelist=[],[]
    for count,prob in enumerate(text_probs):
        if prob.argmax()<101:
            food_filelist.append(file_list[count])
        else:
            caltech_filelist.append(file_list[count])
    f=open(output_path_food,'w')
    f.write('\n'.join(food_filelist))
    f.close()

    f=open(output_path_caltech,'w')
    f.write('\n'.join(caltech_filelist))
    f.close()

def get_4_pseudo_calss(model_clip_path):

    trainset_query_dir='./trainset_clip_query/'
    os.makedirs('./trainset_pseudo_labels/', exist_ok=True)
    output_path_thudog='./trainset_pseudo_labels/Thu-dog-4.txt'
    output_path_animal='./trainset_pseudo_labels/Animal-4.txt'
    output_path_food='./trainset_pseudo_labels/Food-4.txt'
    output_path_caltech='./trainset_pseudo_labels/Caltech-4.txt'
    print('get thudog')
    get_thudog(trainset_query_dir,output_path_thudog,model_clip_path)
    print('get animal')
    get_animal(trainset_query_dir,output_path_thudog,output_path_animal,model_clip_path)
    print('get caltech and food')
    get_food_caltech(trainset_query_dir,output_path_thudog,output_path_animal,output_path_food,output_path_caltech,model_clip_path)

