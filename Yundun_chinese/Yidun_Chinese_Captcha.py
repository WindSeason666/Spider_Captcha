from __future__ import division
from detect.models import Darknet
from detect.utils.utils import *
from detect.utils.datasets import *
from settings import *
from io import BytesIO
import cv2
import base64
import torch
import torchvision
from PIL import Image
from efficientnet_pytorch import EfficientNet
from albumentations.pytorch import ToTensor
import torchvision.transforms as transforms
from albumentations import (Compose, Resize)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = 192
img_save = 'output/now.jpg'

class Predict_chinese():
    def __init__(self):
        self.net = EfficientNet.from_name('efficientnet-b4')
        state_dict = torch.load(efficientnet_b4_pth)
        self.net.load_state_dict(state_dict)
        in_fea = self.net._fc.in_features
        self.net._fc = nn.Linear(in_features=in_fea, out_features=2181, bias=True)
        # self.net = EfficientNet.from_pretrained('efficientnet-b1', num_classes=2181)
        self.net.load_state_dict(torch.load(model_chinese_crop_b4, map_location=device))
        self.net.eval()
        self.transform = transforms.Compose([
            transforms.Lambda(self.albumentations_transform),
        ])
        self.trainset = torchvision.datasets.ImageFolder(root=data_train_path,
                                                    transform=self.transform)

    def strong_aug(self,p=1):
        return Compose([
            Resize(100, 100),
            ToTensor()
        ], p=p)

    def albumentations_transform(self,image, transform=strong_aug(1)):
        if transform:
            image_np = np.array(image)
            augmented = transform(image=image_np)
            image = augmented['image']
        return image

    def predict_chinese(self,img):
        # img = Image.open('data1/test/kk/00b1d9958bf34b9f8ba7d1070cb28c9d_醉.jpg')
        img = self.transform(img).unsqueeze(0)
        pred = self.net(img)
        _, predicted = pred.max(1)
        result = self.trainset.classes[predicted[0]]
        return result


class Captcha_yidun_chinese_position(object):
    def __init__(self, config_path, model_path,names='model/predict_position/target.names',conf_thres=0.3, nms_thres=0.6,save_image=False):
        self.model = self.load_position_model(config_path, model_path) #加载目标检测模型
        self.chinese_model = Predict_chinese()
        self.conf_thres = conf_thres #object confidence threshold
        self.nms_thres = nms_thres  #IOU threshold for NMS
        self.save_image = save_image  #是否保存检测标注后的图片
        self.names = load_classes(names) #加载目标检测类别名字
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
        if os.path.exists('output'):
            pass  # delete output folder
        else:
            os.makedirs('output')  # make new output folder

    def load_position_model(self,config_path, model_path):
        model = Darknet(config_path, img_size=img_size).to(device)
        if model_path.endswith(".weights"):
            model.load_darknet_weights(model_path)
        else:
            model.load_state_dict(torch.load(model_path, map_location=device)['model'])
        model.to(device).eval()
        return model

    def crop_chinese_and_recognize(self,data):
        im = Image.open(img_save)
        region = im.crop((data['x'], data['y'], data['x'] + data['width'], data['y'] + data['height'])).resize((60, 60))
        base64_chinese_single = self.chinese_model.predict_chinese(region)
        return base64_chinese_single

    def image_resize_and_convert(self,image_base64):  # 图片重置大小与二值化处理
        image = base64.b64decode(image_base64)
        img_f = BytesIO(image)
        image = Image.open(img_f)
        out = image.resize((320, 160), Image.ANTIALIAS)  # resize image with high-quality
        out.save(img_save)

    def predict_postion(self, image_path):
        try:
            self.image_resize_and_convert(image_path)
            image_path = 'output/now.jpg'
        except:
            print(f'当前输入：{image_path}')

        im0s = cv2.imread(image_path)
        img = letterbox(im0s, new_shape=img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        with torch.no_grad():
            pred = self.model(img)[0]
            det = non_max_suppression(pred, self.conf_thres, self.nms_thres)[0]
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
            item_list = {}
            for *xyxy, conf, cls in det:
                item = {}
                item_center = {}
                position_one = ('%g ' * 6 ) % (*xyxy, cls, conf)
                position_one_list = [float(i) for i in position_one.split(' ')[:-1]]
                item["x"] = position_one_list[0]
                item["y"] = position_one_list[1]
                item["width"] = position_one_list[2] - position_one_list[0]
                item["height"] = position_one_list[3] - position_one_list[1]
                item["x_center"] = item["x"] + item["width"]/2
                item_center['x'] = int(item["x_center"])
                item["y_center"] = item["y"] + item["height"]/2
                item_center['y'] = int(item["y_center"])
                item["class"] = '0'
                item["acc"] = position_one_list[-1]
                chinese = self.crop_chinese_and_recognize(data=item)
                item_list[chinese] = item_center
                if self.save_image:
                    label = '%s %.2f' % (self.names[int(cls)], conf)
                    plot_one_box(xyxy, im0s, label=label, color=self.colors[int(cls)])
                    cv2.imwrite(f'output/{image_path}', im0s)
            print(f"【易盾：文字点选】识别结果：{item_list}")
            return item_list


if __name__ == '__main__':
      image_path = 'output/now.jpg'
      detector = Captcha_yidun_chinese_position(model_path_chinese_cfg, model_path_chinese,save_image=False)
      position = detector.predict_postion(image_path)




