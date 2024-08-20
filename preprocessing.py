import os
import time
import shutil
import p8_splitTestImageTo4Dataset
import p9_finetuneCaltechAnimalCar
import p10_crop_thudog
import p11_get_thudog_query

def p8():

    print('**********************************************')
    print('p8_splitTestImageTo4Dataset')
    start_time=time.time()
    p8_splitTestImageTo4Dataset.splitTestImageTo4DataSet(TestSetDir,model_clip_path)
    end_time=time.time()
    running_time = end_time-start_time
    print('time cost : %.5f sec' %running_time)

def p9():
    print('**********************************************')
    print('p9_finetuneCaltechAnimalCar')
    start_time=time.time()
    train_dir_caltech = './TrainSet_4_image/Caltech/'
    caltech_model_res_50_path = './model_caltech/caltech_res50.pkl'
    test_imgs_dir_caltech='./TestSetB_5_class/Caltech/'
    test_imgs_dir_animal='./TestSetB_5_class/Animal/'
    test_imgs_dir_car='./TestSetB_5_class/Car/'
    p9_finetuneCaltechAnimalCar.finetune(test_imgs_dir_car,test_imgs_dir_animal,train_dir_caltech,caltech_model_res_50_path,test_imgs_dir_caltech)
    end_time=time.time()
    running_time = end_time-start_time
    print('time cost : %.5f sec' %running_time)

def p10():

    print('**********************************************')
    print('p10_crop_thudog')
    start_time=time.time()
    input_path = './TestSetB_5_class/Thu-dog/'
    output_path = './TestSetB_5_class/Thu-dog-crop/'
    os.makedirs(output_path, exist_ok=True)
    p10_crop_thudog.crop_thudog(input_path, output_path)
    end_time=time.time()
    running_time = end_time-start_time
    print('time cost : %.5f sec' %running_time)

def p11():

    print('**********************************************')
    print('p11_get_thudog_query')
    start_time=time.time()

    query_save_dir = './model_thudog_query/'
    os.makedirs(query_save_dir,exist_ok=True)
    test_img_dir_thudog = './TestSetB_5_class/Thu-dog/'
    test_images_path = ['./TestSetB_5_class/Thu-dog/' + i for i in os.listdir(test_img_dir_thudog)]
    test_images_crop_path = ['./TestSetB_5_class/Thu-dog-crop/' + i for i in os.listdir(test_img_dir_thudog)]
    query_clip_path = query_save_dir + '/thudog_query_clip.npy'
    query_b6_path = query_save_dir + '/thudog_query_b6.npy'
    query_clip_crop_path = query_save_dir + '/thudog_query_crop_clip.npy'
    query_b6_crop_path = query_save_dir + '/thudog_query_crop_b6.npy'
    model_b6_path = './publicModel/efficientnet-b6-c76e70fd.pkl'

    print('get_query_clip')
    p11_get_thudog_query.get_query_clip(model_clip_path, query_clip_path, test_images_path)
    p11_get_thudog_query.get_query_clip(model_clip_path, query_clip_crop_path, test_images_crop_path)

    print('get_query_b6')
    p11_get_thudog_query.get_query_b6(model_b6_path, query_b6_path, test_images_path)
    p11_get_thudog_query.get_query_b6(model_b6_path, query_b6_crop_path, test_images_crop_path)

    end_time=time.time()
    running_time = end_time-start_time
    print('time cost : %.5f sec' %running_time)

def get_Train_4_image(train_dir,train_4_image_list):
    print('**********************************************')
    print('get_Train_4_image')
    for i in open(train_4_image_list).readlines():
        i=i.replace('\n','')
        img=i.split('/')[-1]
        path=i.replace(img,'')
        os.makedirs(path,exist_ok=True)
        shutil.copy(train_dir+path.replace('./TrainSet_4_image','').replace('Caltech','Caltech-101').replace('Food','Food-101')+img,path+img)

TestSetDir='./officalData/TestSetB/' #测试集路径
model_clip_path='./publicModel/ViT-B-32.pkl'

train_dir='./officalData/TrainSet/'
train_4_image_list='./TrainSet_4_image.txt'
get_Train_4_image(train_dir,train_4_image_list)

p8()
p9()
p10()
p11()
