# Jittor 开放域少样本视觉分类赛题

# 1环境配置与目录结构

## 1.1 环境配置

    详见requirements.txt

## 1.2 目录结构
    .
    ├── jCLIP
    │   ├── bpe_simple_vocab_16e6.txt.gz
    │   ├── CLIP.py
    │   ├── __init__.py
    │   ├── mha.py
    │   ├── model_change.py
    │   ├── model.py
    │   └── simple_tokenizer.py
    ├── officalData
    │   ├── TestSetB
    │   └── TrainSet
    ├── TrainSet_4_image
    │   ├── Animal
    │   ├── Caltech
    │   ├── Food
    │   ├── Food-101-trainset-pesudo-label
    │   └── Thu-dog
    ├── trainset_pseudo_labels
    │   ├── Animal-4.txt
    │   ├── Caltech-4.txt
    │   ├── Food-101.txt
    │   ├── Food-4.txt
    │   └── Thu-dog-4.txt
    ├── TestSetB_5_class
    │   ├── Animal
    │   ├── Caltech
    │   ├── Car
    │   ├── Food
    │   └── Thu-dog
    ├── model_caltech
    │   ├── caltech_adapter_f_img.pkl
    │   ├── caltech_cache_values.npy
    │   └── caltech_res50.pkl
    ├── model_food
    │   ├── food_adapter_f_img.pkl
    │   ├── food_cache_values.npy
    │   └── food_CLIP.pkl
    ├── model_thudog_keys
    │   ├── thudog_CLIP_weight.npy
    │   ├── thudog_keys_b6.npy
    │   ├── thudog_keys_CLIP.npy
    │   ├── thudog_keys_trainset_pesudo_b6.npy
    │   ├── thudog_keys_trainset_pesudo_CLIP.npy
    │   ├── thudog_values_b6.npy
    │   ├── thudog_values_CLIP.npy
    │   ├── thudog_values_trainset_pesudo_b6.npy
    │   └── thudog_values_trainset_pesudo_CLIP.npy
    ├── model_thudog_query
    │   ├── thudog_query_b6.npy
    │   ├── thudog_query_CLIP.npy
    │   ├── thudog_query_crop_b6.npy
    │   └── thudog_query_crop_CLIP.npy
    ├── publicModel
    │   ├── efficientnet-b6-c76e70fd.pkl
    │   ├── resnet50.pkl
    │   ├── RN101.pkl
    │   ├── ViT-B-32.pkl
    ├── trainset_b6_query.zip
    ├── trainset_CLIP_query.zip
    ├── train.sh
    ├── train.py
    ├── train.log
    ├── preprocessing.py
    ├── preprocessing.log
    ├── test.py
    ├── test.log
    ├── t0_convert_model.py
    ├── t1_get_trainset_query.py
    ├── t2_get_TrainSet_4_Pseudo_Labels.py
    ├── t3_train_adapter_f_caltech.py
    ├── t4_train_caltech_res50.py
    ├── t5_train_adapter_f_food.py
    ├── t6_get_TrainSet_Food_Pseudo_Labels.py
    ├── t7_train_food_pesudo_label_CLIP.py
    ├── p8_splitTestImageTo4Dataset.py
    ├── p9_finetuneCaltechAnimalCar.py
    ├── p10_crop_thudog.py
    ├── p11_get_thudog_query.py
    ├── i12_infer_food.py
    ├── i13_infer_animal.py
    ├── i14_infer_caltech.py
    ├── i15_infer_thudog.py
    ├── i16_infer_car.py
    ├── i17_infer_summary.py
    ├── result_animal.txt
    ├── result_caltech.txt
    ├── result_car.txt
    ├── result_food.txt
    ├── result_thudog.txt
    ├── result.txt
    ├── TrainSet_4_image.txt
    ├── classes_b.txt
    ├── requirements.txt
    ├── EfficientNetModel.py
    ├── EfficientNetUtils.py
    ├── query2img.json
    ├── car_cate.json
    └── food_categories.json

# 2 方法的详细思路
## 2.1 训练阶段
### 2.1.1 训练集图像粗分类
    使用CLIP[1]模型对训练集中所有图像进行推理,把所有训练集图像分为Food、Caltech、Thu-dog、Animal四类,对应代码为t1_get_trainset_query.py（用于提取图像特征）、t2_get_TrainSet_4_Pseudo_Labels.py（用于把所有训练集图像分为四类）,后续将对这四类进一步分类.
    
    特征提取后存放路径为：
    ├── trainset_b6_query
    ├── trainset_CLIP_query

    推理结果存放路径为:
    ├── trainset_pseudo_labels
    │   ├── Animal-4.txt
    │   ├── Caltech-4.txt
    │   ├── Food-4.txt
    │   └── Thu-dog-4.txt

### 2.1.2 提取每个类别的4张训练图像
    获取每个类别的4张图像,对应代码为train.py中的get_Train_4_image(train_dir,train_4_image_list),其中train_dir为官方训练集路径,在本项目中路径为
    ├── officalData
    │   └── TrainSet
    
    train_4_image_list为每个类别4张图像的列表,存储在TrainSet_4_image.txt中,get_Train_4_image(train_dir,train_4_image_list)函数读取train_4_image_list文件,提取每类4张训练图像,提取后存放路径为:
    ├── TrainSet_4_image
    │   ├── Animal
    │   ├── Caltech
    │   ├── Food
    │   └── Thu-dog

### 2.1.3 训练Caltech分类模型
    基于Tip-Adapter-F[2]训练Caltech分类模型,对应代码t3_train_adapter_f_caltech.py；基于ResNet50[3]训练Caltech分类模型,对应代码t4_train_caltech_res50.py.模型存放路径为:
    model_caltech
    │   ├── caltech_adapter_f_img.pkl
    │   ├── caltech_cache_values.npy
    │   └── caltech_res50.pkl；

### 2.1.4 训练Food分类模型
    基于Tip-Adapter-F训练Food分类模型,对应代码t5_train_adapter_f_food.py,模型存放路径为:
    ├── model_food
    │   ├── food_adapter_f_img.pkl
    │   ├── food_cache_values.npy
    
    使用训练好的模型food_adapter_f_img.pkl,对2.1.1中获取的Food-4.txt文件进行推理,获取食物细粒度类别,然后提取图像用于进一步训练,对应代码为t6_get_TrainSet_Food_Pseudo_Labels.py,提取后图像存放于:
    ├── TrainSet_4_image
    │   ├── Food-101-trainset-pesudo-label
    
    在CLIP模型的图像编码器后面添加全连接层,使用Food-101-trainset-pesudo-label训练该模型,对应代码为t7_train_food_pesudo_label_CLIP.py,模型存放路径为:
    ├── model_food
    │   └── food_CLIP.pkl

## 2.2 测试图像预处理阶段
### 2.2.1 测试图像粗分类
    使用CLIP模型对所有测试图像进行推理,分为Food、Caltech、Thu-dog、Animal、Car五类,对应代码为p8_splitTestImageTo4Dataset.py,本项目中测试图像存放路径为:
    ├── officalData
    │   ├── TestSetB
    
    推理后提取测试图像,存放路径为:
    ├── TestSetB_5_class
    │   ├── Animal
    │   ├── Caltech
    │   ├── Car
    │   ├── Food
    │   └── Thu-dog
    
    由于Animal、Car和Caltech数据集有类别重叠,使用训练好的caltech模型对Animal和Car的粗分类结果进行推理,对推理结果置信度较高的图像再分为Caltech类,对应代码为p9_finetuneCaltechAnimalCar.py
### 2.2.2 Thu-dog类别预处理
    基于Efficientnet-B6[4]预训练模型对图像不同区域的注意力图,裁剪掉Thu-dog类别图像的背景区域,留下主体区域,对应代码为:p10_crop_thudog.py,裁剪后图像的存放路径为:
    ├── TestSetB_5_class
    │   └── Thu-dog-crop
    
    分别使用CLIP、Efficientnet-B6提取Thu-dog、Thu-dog-crop的图像特征,用于后续推理,提取后特征存放路径为:
    ├── model_thudog_query
    │   ├── thudog_query_b6.npy
    │   ├── thudog_query_CLIP.npy
    │   ├── thudog_query_crop_b6.npy
    │   └── thudog_query_crop_CLIP.npy

# 2.3 推理阶段
## 2.3.1 推理Food类别
    使用2.1.4训练得到的Food分类模型对2.2.2得到的Food类别图像进行推理,获取结果result_food.txt,对应代码为i12_infer_food.py
## 2.3.2 推理Animal类别
    使用Tip-Adapter模型[2]对2.2.2得到的Animal类别图像进行推理,获取结果result_animal.txt,对应代码为i13_infer_animal.py
## 2.3.3 推理Caltech类别
    使用2.1.3训练得到的Caltech分类模型对2.2.2得到的Caltech类别图像进行推理,获取结果result_caltech.txt,对应代码为i14_infer_caltech.py
## 2.3.4 推理Thudog类别
    使用Tip-Adapter模型对2.1.1获取的Thu-dog类别的图像进行推理,获取训练集中Thu-dog类别的伪标签。基于获取的伪标签和Thu-dog中每类的4张图像,使用Tip-Adapter模型对2.2.2得到的Thudog类别图像进行推理,获取结果result_thudog.txt,对应代码为i15_infer_thudog.py
## 2.3.5 推理Car类别
    使用Tip-Adapter模型对2.2.2得到的Car类别图像进行推理,获取结果result_car.txt,对应代码为i16_infer_car.py
## 2.3.6 汇总推理结果
    对result_food.txt,result_animal.txt,result_thudog.txt,result_car.txt,result_car.txt文件进行汇总,得到最终结果result.txt

# 4 训练

    在2080ti显卡上，各步骤运行时间见train.log
    训练命令：sh train.sh
    

# 5 测试

    (1)对测试图像进行预处理,在2080ti显卡上,各步骤运行时间见preprocessing.log
    预处理命令：python preprocessing.py

    (2)进行推理,在2080ti显卡上,各步骤运行时间见test.log
    测试命令：python test.py

# 6 使用的预训练模型种类
   
    1 OpenAI官方预训练的ViT-B/32版本的CLIP模型,地址：https://openaipublic.azureedge.net/CLIP/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
    2 OpenAI官方预训练的ResNet101版本的CLIP模型,地址：https://openaipublic.azureedge.net/CLIP/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt
    3 在ImageNet-1K上预训练的efficientnet-B6模型,地址：https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b6-c76e70fd.pth
    4 在ImageNet-1K上预训练的ResNet50模型,地址：https://cg.cs.tsinghua.edu.cn/jittor/assets/build/checkpoints/resnet50.pkl

# 7 最终的模型参数量之和
    
    OpenAI官方预训练的ViT-B/32版本的CLIP模型:144.28MB
    OpenAI官方预训练的ResNet101版本的CLIP模型:114.25MB
    Clatech类别的ResNet50模型:22.65MB
    Clatech类别的adapter模型:0.17MB
    Food类别在CLIP视觉编码器基础上微调后的模型:84.08MB
    Food类别的adapter模型:0.25MB
    efficientnet-B6模型:41.26MB
    
    最终参数量之和：406.94MB

# 8 联系方式
    qq:205992690
    手机:18617521225

# 参考文献

    [1]Radford A, Kim J W, Hallacy C, et al. Learning transferable visual models from natural language supervision[C]//International conference on machine learning. PMLR, 2021: 8748-8763.
    [2] Zhang R, Fang R, Zhang W, et al. Tip-adapter: Training-free CLIP-adapter for better vision-language modeling[J]. arXiv preprint arXiv:2111.03930, 2021.
    [3] He K, Zhang X, Ren S, et al. Deep residual learning for image recognition[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 770-778.
    [4]Tan M, Le Q. Efficientnet: Rethinking model scaling for convolutional neural networks[C]//International conference on machine learning. PMLR, 2019: 6105-6114.
