import os
import time
import shutil
import argparse

import t1_get_trainset_query
import t2_get_TrainSet_4_Pseudo_Labels
import t3_train_adapter_f_caltech
import t4_train_caltech_res50
import t5_train_adapter_f_food
import t6_get_TrainSet_Food_Pseudo_Labels
import t7_train_food_pesudo_label_clip

import jittor as jt
jt.flags.use_cuda = 1

# 创建ArgumentParser对象
model_clip_path='./publicModel/ViT-B-32.pkl'

parser = argparse.ArgumentParser()
parser.add_argument('train_dir', type=str)
parser.add_argument('train_4_image_list', type=str)
args = parser.parse_args()
train_dir = args.train_dir.replace('\r','')
train_4_image_list = args.train_4_image_list.replace('\r','')

def get_Train_4_image(train_dir,train_4_image_list):
    for i in open(train_4_image_list).readlines():
        i=i.replace('\n','')
        img=i.split('/')[-1]
        path=i.replace(img,'')
        os.makedirs(path,exist_ok=True)
        shutil.copy(train_dir+path.replace('./TrainSet_4_image','').replace('Caltech','Caltech-101').replace('Food','Food-101')+img,path+img)

def t0():
    #获取训练集每类的4张图像
    print('**********************************************')
    print('get_Train_4_image')
    get_Train_4_image(train_dir,train_4_image_list)

def t1():

    #获取训练集图像所有特征，用于后续半监督学习，制作伪标签
    print('**********************************************')
    print('t1_get_trainset_query')
    model_clip_path="./publicModel/ViT-B-32.pkl"
    model_b6_path='./publicModel/efficientnet-b6-c76e70fd.pkl'

    start_time=time.time()
    output_path_trainset_query_clip='./trainset_clip_query/'
    output_path_trainset_query_b6='./trainset_b6_query/'
    imgs=t1_get_trainset_query.get_query2img_dict(train_dir)

    print('get_trainset_clip_query')
    t1_get_trainset_query.get_trainset_clip_query(imgs,output_path_trainset_query_clip,model_clip_path)
    end_time=time.time()
    running_time = end_time-start_time
    print('time cost : %.5f sec' %running_time)

    print('get_trainset_b6_query')
    start_time=time.time()
    t1_get_trainset_query.get_trainset_b6_query(imgs,output_path_trainset_query_b6,model_b6_path)
    end_time=time.time()
    running_time = end_time-start_time
    print('time cost : %.5f sec' %running_time)

def t2():
    #把训练集所有图像分成4类
    print('**********************************************')
    print('t2_get_TrainSet_4_Pseudo_Labels')
    start_time=time.time()
    t2_get_TrainSet_4_Pseudo_Labels.get_4_pseudo_calss(model_clip_path)
    end_time=time.time()
    running_time = end_time-start_time
    print('time cost : %.5f sec' %running_time)

def t3():
    #训练caltech_adapter_f模型
    print('**********************************************')
    print('t3_train_adapter_f_caltech')
    start_time=time.time()
    train_dir = './TrainSet_4_image/Caltech/'
    t3_train_adapter_f_caltech.train_adapter_f_caltech(train_dir)
    end_time=time.time()
    running_time = end_time-start_time
    print('time cost : %.5f sec' %running_time)

def t4():
    #训练caltech_res50模型
    print('**********************************************')
    print('t4_train_caltech_res50')
    start_time = time.time()
    train_dir_caltech = './TrainSet_4_image/Caltech/'
    t4_train_caltech_res50.train_caltch_res50(train_dir_caltech)
    end_time=time.time()
    running_time = end_time-start_time
    print('time cost : %.5f sec' %running_time)

def t5():
    #训练food_adapter_f模型
    print('**********************************************')
    print('t5_train_adapter_f_food')
    start_time=time.time()
    train_dir_food = './TrainSet_4_image/Food/'
    t5_train_adapter_f_food.train_adapter_f_food(train_dir_food,model_clip_path)
    end_time=time.time()
    running_time = end_time-start_time
    print('time cost : %.5f sec' %running_time)

def t6():
    #使用半监督学习，推理通过t2_get_TrainSet_4_Pseudo_Labels获取的食物类别的伪标签
    print('**********************************************')
    print('t6_get_TrainSet_Food_Pseudo_Labels')
    start_time = time.time()
    adapter_f_model_path = './model_food/food_adapter_f_img.pkl'
    cache_calue_path = './model_food/food_cache_values.npy'
    t6_get_TrainSet_Food_Pseudo_Labels.get_trainset_food_pseudo_label(model_clip_path,adapter_f_model_path,cache_calue_path)
    t6_get_TrainSet_Food_Pseudo_Labels.get_trainset_pesudo_labels_images()
    end_time=time.time()
    running_time = end_time-start_time
    print('time cost : %.5f sec' %running_time)

def t7():
    #使用获取的食物伪标签，训练clip的image encoder部分，训练食物分类器
    print('**********************************************')
    print('t7_train_food_pesudo_label_clip')
    start_time = time.time()
    train_dir_food_pesudo_label= './TrainSet_4_image/Food-101-trainset-pesudo-label/'
    t7_train_food_pesudo_label_clip.train(model_clip_path,train_dir_food_pesudo_label)
    end_time=time.time()
    running_time = end_time-start_time
    print('time cost : %.5f sec' %running_time)

t0()
t1()
t2()
t3()
t4()
t5()
t6()
t7()