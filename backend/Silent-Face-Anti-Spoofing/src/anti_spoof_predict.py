# -*- coding: utf-8 -*-
# @Time : 20-6-9 上午10:20
# @Author : zhuying
# @Company : Minivision
# @File : anti_spoof_predict.py
# @Software : PyCharm

import os
import cv2
import math
import torch
import numpy as np
import torch.nn.functional as F


from src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2,MiniFASNetV1SE,MiniFASNetV2SE
from src.data_io import transform as trans
from src.utility import get_kernel, parse_model_name

MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE':MiniFASNetV1SE,
    'MiniFASNetV2SE':MiniFASNetV2SE
}


class Detection:
    def __init__(self):
        resource_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "resources",
            "detection_model",
        )
        caffemodel = os.path.join(resource_dir, "Widerface-RetinaFace.caffemodel")
        deploy = os.path.join(resource_dir, "deploy.prototxt")
        self.detector = cv2.dnn.readNetFromCaffe(deploy, caffemodel)
        self.detector_confidence = 0.6

    def get_bbox(self, img):
        height, width = img.shape[0], img.shape[1]
        aspect_ratio = width / height
        if img.shape[1] * img.shape[0] >= 192 * 192:
            img = cv2.resize(img,
                             (int(192 * math.sqrt(aspect_ratio)),
                              int(192 / math.sqrt(aspect_ratio))), interpolation=cv2.INTER_LINEAR)

        blob = cv2.dnn.blobFromImage(img, 1, mean=(104, 117, 123))
        self.detector.setInput(blob, 'data')
        out = self.detector.forward('detection_out').reshape(-1, 7)
        if out.size == 0:
            return None
        confidences = np.nan_to_num(out[:, 2], nan=-np.inf)
        max_conf_index = np.argmax(confidences)
        if confidences[max_conf_index] < self.detector_confidence:
            return None
        left = int(out[max_conf_index, 3] * width)
        top = int(out[max_conf_index, 4] * height)
        right = max(0, min(width, int(out[max_conf_index, 5] * width)))
        bottom = max(0, min(height, int(out[max_conf_index, 6] * height)))
        left = max(0, min(width - 1, left))
        top = max(0, min(height - 1, top))
        bbox = [left, top, right - left, bottom - top]
        if bbox[2] <= 0 or bbox[3] <= 0:
            return None
        return bbox


class AntiSpoofPredict(Detection):
    def __init__(self, device_id):
        super(AntiSpoofPredict, self).__init__()
        self.device = torch.device("cuda:{}".format(device_id)
                                   if torch.cuda.is_available() else "cpu")
        self.models = {}

    def load_model(self, model_path):
        model_path = os.path.abspath(model_path)
        if model_path in self.models:
            return self.models[model_path]

        model_name = os.path.basename(model_path)
        h_input, w_input, model_type, _ = parse_model_name(model_name)
        kernel_size = get_kernel(h_input, w_input)
        model = MODEL_MAPPING[model_type](conv6_kernel=kernel_size).to(self.device)

        state_dict = torch.load(model_path, map_location=self.device)
        first_layer_name = next(iter(state_dict))
        if first_layer_name.startswith('module.'):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                name_key = key[7:]
                new_state_dict[name_key] = value
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)
        model.eval()
        self.models[model_path] = model
        return model

    def predict(self, img, model_path):
        test_transform = trans.Compose([
            trans.ToTensor(),
        ])
        img = test_transform(img)
        img = img.unsqueeze(0).to(self.device)
        model = self.load_model(model_path)
        with torch.no_grad():
            result = model(img)
            result = F.softmax(result, dim=1).cpu().numpy()
        return result








