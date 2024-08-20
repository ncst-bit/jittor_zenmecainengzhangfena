import os
import cv2
import numpy as np
import jittor as jt
from PIL import Image
import jittor.transform as transforms
from EfficientNetModel import EfficientNet as EfficientNet_jittor

jt.flags.use_cuda = 1

def returnCAM(feature_conv, weight_softmax, class_idx):
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        feature_conv = feature_conv.reshape((nc, h * w))
        cam = weight_softmax[idx].dot(
            feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam_img = (cam - cam.min()) / (cam.max() - cam.min())
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cam_img)
    return output_cam

def crop_roi(image, mask, image_name, index, output_path):
    heigth, width, _ = image.shape
    mask[mask > 120] = 255
    mask[mask < 120] = 0

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    x, y, w, h = cv2.boundingRect(max_contour)
    center_x = x + w // 2
    center_y = y + h // 2

    for i in [1]:
        w1, h1 = int(w * i), int(h * i)
        w1, h1 = max(w1,h1), max(w1,h1),
        x1, y1 = int(center_x - w1 // 2), int(center_y - h1 // 2)
        roi = image[max(0, y1):min(y1 + h1, heigth), max(0, x1):min(width, x1 + w1)]
        result_name = output_path + image_name
        cv2.imwrite(result_name, roi)

class EfficientNetFeatureExtractor(jt.nn.Module):
    def __init__(self, original_model):
        super(EfficientNetFeatureExtractor, self).__init__()
        self.stem = original_model._conv_stem
        self.bn0 = original_model._bn0
        self.blocks = original_model._blocks
        self.head_conv = original_model._conv_head
        self.head_bn = original_model._bn1

    def execute(self, x):
        x = self.stem(x)
        x = self.bn0(x)
        for block in self.blocks:
            x = block(x)
        x = self.head_conv(x)
        x = self.head_bn(x)
        return x

def crop_thudog(input_path, output_path):

    model = EfficientNet_jittor.from_name('efficientnet-b6')
    model.load_state_dict(jt.load('./publicModel/efficientnet-b6-c76e70fd.pkl'))
    feature_extractor = EfficientNetFeatureExtractor(model)
    fc_weights = model.state_dict()['_fc.weight'].cpu().numpy()
    model.eval()
    feature_extractor.eval()

    preprocess = transforms.Compose([
        transforms.Resize((528, 528)),
        transforms.ImageNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for image_name in os.listdir(input_path):
        if image_name not in os.listdir(output_path):
            image_path = input_path + image_name
            image = Image.open(image_path).convert('RGB')
            input_tensor = preprocess(image)
            input_batch = jt.Var(input_tensor).unsqueeze(0)
            with jt.no_grad():
                predict = model(input_batch).detach().numpy()[0][150:270]
                features = feature_extractor(input_batch)
                top_5_indices = np.argsort(predict)[-100:][::-1]
                targets = []
                for i in top_5_indices:
                    if len(targets) < 2:
                        targets.append(i + 150)
                CAMs = returnCAM(features.detach().numpy(), fc_weights, targets)
                img = cv2.imread(image_path)
                height, width, _ = img.shape
                mask = np.zeros_like(CAMs[0])
                for i in range(len(targets)):
                    mask = mask + CAMs[i]
                    mask = mask + CAMs[i]
                crop_roi(img, cv2.resize(np.array(mask).astype(np.uint8), (width, height)), image_name, i, output_path)
