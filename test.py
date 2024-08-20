import os
import time

import i12_infer_food
import i13_infer_animal
import i14_infer_caltech
import i15_infer_thudog
import i16_infer_car
import i17_infer_summary

def i12():
    print('**********************************************')
    print('i12_infer_food')
    start_time=time.time()
    model_clip_rn101_path="./publicModel/RN101.pkl"
    test_imgs_dir = './TestSetB_5_class/Food/'
    food_adapter_f_model_path = './model_food/food_adapter_f_img.pkl'
    food_cache_calue_path = './model_food/food_cache_values.npy'
    i12_infer_food.infer(test_imgs_dir,model_clip_path,model_clip_rn101_path,food_adapter_f_model_path,food_cache_calue_path)
    end_time=time.time()
    running_time = end_time-start_time
    print('time cost : %.5f sec' %running_time)

def i13():
    print('**********************************************')
    print('i13_infer_animal')
    start_time=time.time()
    train_dir_animal='./TrainSet_4_image/Animal/'
    test_dir_animal='./TestSetB_5_class/Animal/'
    i13_infer_animal.infer(train_dir_animal,test_dir_animal,model_clip_path)
    end_time=time.time()
    running_time = end_time-start_time
    print('time cost : %.5f sec' %running_time)

def i14():
    print('**********************************************')
    print('i14_infer_caltech')
    start_time=time.time()
    test_imgs_dir_caltech = './TestSetB_5_class/Caltech/'
    train_dir_caltech = './TrainSet_4_image/Caltech/'
    caltech_model_res_50_path = './model_caltech/caltech_res50.pkl'
    caltech_adapter_f_model_path = './model_caltech/caltech_adapter_f_img.pkl'
    caltech_cache_calue_path = './model_caltech/caltech_cache_values.npy'
    i14_infer_caltech.infer(model_clip_path,test_imgs_dir_caltech,train_dir_caltech,caltech_model_res_50_path,caltech_adapter_f_model_path,caltech_cache_calue_path)
    end_time=time.time()
    running_time = end_time-start_time
    print('time cost : %.5f sec' %running_time)

def i15():

    print('**********************************************')
    print('i15_infer_thudog')
    start_time=time.time()
    model_b6_path = './publicModel/efficientnet-b6-c76e70fd.pkl'
    train_dir_thudog = './TrainSet_4_image/Thu-dog/'
    test_img_dir_thudog = './TestSetB_5_class/Thu-dog/'
    i15_infer_thudog.infer(model_clip_path, model_b6_path, train_dir_thudog, test_img_dir_thudog)
    end_time=time.time()
    running_time = end_time-start_time
    print('time cost : %.5f sec' %running_time)

def i16():

    print('**********************************************')
    print('i16_infer_car')
    start_time=time.time()
    model_clip_rn101_path = "./publicModel/RN101.pkl"
    test_imgs_dir_car = './TestSetB_5_class/Car/'
    i16_infer_car.infer(test_imgs_dir_car, model_clip_path, model_clip_rn101_path)
    end_time=time.time()
    running_time = end_time-start_time
    print('time cost : %.5f sec' %running_time)

def i17():

    print('**********************************************')
    print('i16_infer_summary')
    animal_path = 'result_animal.txt'
    caltech_path = 'result_caltech.txt'
    food_path = 'result_food.txt'
    thudog_path = 'result_thudog.txt'
    car_path = 'result_car.txt'
    i17_infer_summary.infer(animal_path, caltech_path, food_path, thudog_path, car_path)

model_clip_path='./publicModel/ViT-B-32.pkl'
os.system("unzip trainset_b6_query.zip")
os.system("unzip trainset_clip_query.zip")
i12()
i13()
i14()
i15()
i16()
i17()
