import flask
import logging
import base64

from PIL import Image
from io import BytesIO
from flask import request, Flask
from logging.handlers import TimedRotatingFileHandler

from utils.datasets import *
from models import *

app = Flask(__name__)
### LOGGING ###
#formatter = logging.Formatter("[$(asctime)s] %(message)s")
#handler = TimedRotatingFileHandler("/var/log/flask/app/deploy.log", when='midnight', interval=1, backupCount=5)
#handler.setLevel(logging.INFO)
#handler.setFormatter(formatter)
#app.logger.addHandler(handler)

### Loading Pytorch MODEL ###
print("Loading PyTorch YOLOv3 model and Flask starting server ... ")

only_cpu = False
img_size = 416
cfg = "cfg/fashion.cfg"
weights = './weights/exp2/best.pt'

device = torch_utils.select_device(device='cpu' if only_cpu else '')
model = Darknet(cfg, img_size)
if weights.endswith('.pt'):  # pytorch format
    model.load_state_dict(torch.load(weights, map_location=device)['model'])
else:  # darknet format
    _ = load_darknet_weights(model, weights)

model.to(device).eval()
###
def isBase64(s):
    try:
        return base64.b64encode(base64.b64decode(s)) == s
    except Exception:
        return False

def transpose_img(img, img_size):
    # Padded resize
    img, *_ = letterbox(img, new_shape=img_size)
    # Normalize RGB
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32)
    img /= 255.0  # 0-255 to 0-1
    return img

# 사전 스크린 이후 가장 큰 박스 선택
def choose_big_one(det, img_width, img_height, type):
    size = []
    cls_list = []
    conf_list = []
    xyxy_list = []

    size_threshold = 1 / 14
    boxes = len(det)
    for *xyxy, conf, _, cls in det:
        xyxy = [int(x) for x in xyxy]
        dimen = abs(xyxy[2] - xyxy[0]) * abs(xyxy[3] - xyxy[1])

        if type == 'screen':
            if cls in (4, 6, 7, 8):  # 4 bag, 6 Headwear, 7 Acc, 8 Innerwear # 1차 제거
                boxes -= 1
                continue
            else:  # 2차 제거 (박스 크기)
                spatial_ratio = dimen / (img_width * img_height)

                if spatial_ratio < size_threshold:
                    boxes -= 1
                    continue
                else:
                    size.append(dimen)
                    xyxy_list.append(xyxy)
                    conf_list.append(conf)
                    cls_list.append(cls)

    if boxes == 0:  # 박스가 전혀 존재하지않음
        return 0, 0, 0, boxes

    else:  # 박스가 하나라도 존재
        max_idx = size.index(max(size))
        max_xyxy = xyxy_list[max_idx]
        conf = conf_list[max_idx]
        cls = cls_list[max_idx]
        return max_xyxy, conf, cls, boxes


def calc_for_abf(img, max_xyxy, desire_point):
    # numpy type이 uint8이어야 변환됨
    img = Image.fromarray(img.astype('uint8'))
    # PIL Image size (height, width)
    img_w, img_h = (img.size[1], img.size[0])

    s_mask_width, s_mask_height = 130, 166

    # x,y,x,y coordinate & box size
    x1, y1, x3, y3 = max_xyxy
    box_width = x3 - x1
    box_height = y3 - y1

    # calculate padding ratio
    pad_ratio = (s_mask_width - desire_point) / desire_point
    # calculate padding width
    padded_width = x1 + box_width + (box_width * pad_ratio)

    lt_point = (x1, y1)
    padded_rb_point = (padded_width, img_h)

    img_cropped_w, img_cropped_h = (padded_rb_point[0] - lt_point[0]), (padded_rb_point[1] - lt_point[1])

    resize_ratio = s_mask_width / img_cropped_w

    img_resized_w, img_resized_h = s_mask_width, int(img_cropped_h * resize_ratio)

    if padded_width <= img_w:
        border_x = s_mask_width + 1

        if img_resized_h > s_mask_height:
            border_y = s_mask_height + 1
        else:
            border_y = img_resized_h
    else:
        border_x = round((img_w - x1) * resize_ratio) - 2

        if img_resized_h > s_mask_height:
            border_y = s_mask_height + 1
        else:
            border_y = img_resized_h

    border_points = [(border_x, 0), (0, border_y)]

    if padded_width / img_h > 1.15:
        return 0
    return lt_point, padded_rb_point, resize_ratio, border_points

def predict_img(img, type):
    data = 'data/fashion.data'
    classes = load_classes(parse_data_cfg(data)['names'])

    input_img = transpose_img(img, img_size=416)

    input_img = torch.from_numpy(input_img).to(device)
    if input_img.ndimension() == 3:
        input_img = input_img.unsqueeze(0)

    pred, _ = model(input_img)
    # Detections per image
    for i, det in enumerate(non_max_suppression(pred, conf_thres=0.5, nms_thres=0.3)):
        img_height, img_width = input_img.shape[2:]

        if det is not None and len(det):
            det[:, :4] = scale_coords(input_img.shape[2:], det[:, :4], img.shape).round()

            # 가장 큰 박스 선택 & 사전 스크린 P1,P2
            max_xyxy, conf, cls, boxes = choose_big_one(det, img_width, img_height, type)
            
            x1,y1,x3,y3 = max_xyxy
            box_width = x3-x1
            box_height = y3-y1

            if boxes == 0:
                return 0
            else:
                # ABF
                if type == 'abf':
                    result = calc_for_abf(img, max_xyxyx, desire_point=90)
                    if result == 0:
                        return 0
                    lt_point, padded_rb_point, resize_ratio, border_points = result
                    return lt_point, padded_rb_point, resize_ratio, border_points, classes[int(cls)]

                # POST SCREEN
                if type == 'screen':
                    box_cx_norm = (x1 + (box_width / 2)) / img_width
                    box_cy_norm = (y1 + (box_height / 2)) / img_height
                    box_center_point_norm = (box_cx_norm, box_cy_norm)
                    return boxes, classes[int(cls)], box_center_point_norm
        # No Detections
        else:
            return 0

def confirm_request(request):
    torch.cuda.empty_cache()
    data = request.data
    if isBase64(data):  # Base64
        img_bytes = base64.b64decode(data)
        img = Image.open(BytesIO(img_bytes))

    elif type(data) == bytes:  # Bytes
        img = Image.open(BytesIO(data))
    else:
        print(type(data))
        return 'U'
    return img

@app.route('/v1/auto_banner', methods=['POST','GET'])
def automatic_banner():
    if request.method == 'POST':
        img = confirm_request(request)
        if img == 'U':
            return flask.jsonify({
                "status": 'unidentified type of data'
            })
        else:
            img = np.asarray(img, dtype=np.float32)
            # 2d image -> 3d image
            if len(img.shape) == 2:
                img = np.stack((img,) * 3, axis=-1)
            ### ABF ###
            result = predict_img(img, type = 'abf')
            # NO BOX DETECTED
            if result == 0:
                return flask.jsonify({
                    "status": 'F'
                })
            # ANY BOX DETECTED
            else:
                lt_point, padded_rb_point, resize_ratio, border_points, cls = result
                return flask.jsonify({
                    "status": 'S',
                    "class": cls,
                    "lt_point": lt_point,
                    "padded_rb_point": padded_rb_point,
                    "resize_ratio": resize_ratio,
                    "border_points": border_points
                })
    else:  # GET
        return flask.jsonify({
            "status": 'use POST request'
        })

@app.route('/v1/content_check', methods=['POST', 'GET'])
def post_screen():
    if request.method == 'POST':
        img = confirm_request(request)
        if img == 'U':
            return flask.jsonify({
                "status": 'unidentified type of data'
            })
        else:
            img = np.asarray(img, dtype=np.float32)
            # 2d image -> 3d image
            if len(img.shape) == 2:
                img = np.stack((img,) * 3, axis=-1)

            ### POST SCREENING ###
            result = predict_img(img, type = 'screen')
            # NO BOX DETECTED
            if result == 0:
                return flask.jsonify({
                    "status": 'F'
                })
            # ANY BOX DETECTED
            else:
                boxes, cls, box_center_point_norm = result
                return flask.jsonify({
                    "status": 'S',
                    "class": cls,
                    "center_norm": box_center_point_norm
                })
    else:  # GET
        return flask.jsonify({
            "status": 'use POST request'
        })


if __name__ == '__main__':
    app.run()

