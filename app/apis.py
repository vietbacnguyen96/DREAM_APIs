from app import app
from flask import Response, request, jsonify, redirect, url_for, send_file, render_template


import os
import base64
import requests
import numpy as np
import cv2
from PIL import Image
import datetime

import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torch.nn.functional as F
import torchvision.transforms as transforms

from app.backbone import Backbone
from app.vision.ssd.config.fd_config import define_img_size
from app.vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor

from app.arcface_torch.backbones import get_model
from app.arcface_torch.branch_util import *


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('\nUsing ', device, '\n')
input_size=[112, 112]
transform = transforms.Compose(
        [
            transforms.Resize(
                [int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)],
            ),  # smaller side resized
            transforms.CenterCrop([input_size[0], input_size[1]]),
            # transforms.Resize([input_size[0], input_size[1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ],
)

backbone = Backbone(input_size)
backbone.load_state_dict(torch.load('app/ms1mv3_arcface_r50_fp16/backbone_ir50_ms1m_epoch120.pth', map_location=torch.device("cpu")))
backbone.to(device)
backbone.eval()


model_dream = Branch(feat_dim=512)
# model.cuda()
checkpoint = torch.load('./app/arcface_torch/checkpoint_512.pth')
model_dream.load_state_dict(checkpoint['state_dict'])
model_dream.eval()

from utils.TFLiteFaceAlignment import * 
from utils.TFLiteFaceDetector import * 
from utils.functions import *

path = "./"

fd = UltraLightFaceDetecion(path + "utils/weights/RFB-320.tflite", conf_threshold=0.98)
fa = CoordinateAlignmentModel(path + "utils/weights/coor_2d106.tflite")

def loadBase64Img(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def load_image(img):
	exact_image = False; base64_img = False; url_img = False

	if type(img).__module__ == np.__name__:
		exact_image = True

	elif len(img) > 11 and img[0:11] == "data:image/":
		base64_img = True

	elif len(img) > 11 and img.startswith("http"):
		url_img = True

	#---------------------------

	if base64_img == True:
		img = loadBase64Img(img)

	elif url_img:
		img = np.array(Image.open(requests.get(img, stream=True).raw))

	elif exact_image != True: #image path passed as input
		if os.path.isfile(img) != True:
			raise ValueError("Confirm that ",img," exists")

		img = cv2.imread(img)

	return img


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route("/dream", methods=['POST'])
def facerec_DREAM():

    req = request.get_json()
    img_input = ""
    if "img" in list(req.keys()):
        img_input = req["img"]

    validate_img = False
    if len(img_input) > 11 and img_input[0:11] == "data:image/":
        validate_img = True

    if validate_img != True:
        return jsonify({"result": {'message': 'Vui lòng truyền ảnh dưới dạng Base64'}}), 400
    
    orig_image = load_image(img_input)

    # Step 1: Get a face from current frame.
    orig_image = cv2.resize(orig_image, (600, 400), interpolation = cv2.INTER_CUBIC)
    temp_boxes, _ = fd.inference(orig_image)

    # Find landmarks of each face
    landmarks = fa.get_landmarks(orig_image, temp_boxes)
    feats = []

    for bbox_I, landmark_I in zip(temp_boxes, landmarks):
        bbox_I = [int(number) for number in bbox_I]
        x1, y1, x2, y2 = bbox_I
        now = round(datetime.datetime.now().timestamp() * 1000)
        cv2.imwrite('detectedFaces/' + str(now) + '.jpg', orig_image[y1: y2, x1: x2])
        roll, pitch, yaw, _ = estimatePose(orig_image, landmark_I)

        with torch.no_grad():
            embedding_I = F.normalize(backbone(transform(Image.fromarray(orig_image[y1: y2, x1: x2])).unsqueeze(0).to(device))).cpu()
        yaw = np.zeros([1, 1])
        yaw[0,0] = norm_angle(float(yaw))
        original_embedding_tensor = np.expand_dims(embedding_I.detach().cpu().numpy(), axis=0)
        feature_original = torch.autograd.Variable(torch.from_numpy(original_embedding_tensor.astype(np.float32)))
        yaw = torch.autograd.Variable(torch.from_numpy(yaw.astype(np.float32)))

        new_embedding = model_dream(feature_original, yaw)
        new_embedding = new_embedding.to('cpu').data.numpy()
        embedding_I = new_embedding[0, :].tolist()[0]
        feats.append(embedding_I)
    
    return jsonify({'result': {"feats": feats}}), 200

