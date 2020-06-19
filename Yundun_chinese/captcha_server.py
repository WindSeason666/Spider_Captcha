#-*-coding:utf-8-*-
from flask import request
from flask import Flask
from settings import *
import logging
import json
from Yidun_Chinese_Captcha import Captcha_yidun_chinese_position

app = Flask(__name__)

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

CH = Captcha_yidun_chinese_position(model_path_chinese_cfg, model_path_chinese,save_image=False)

@app.route('/captcha', methods=['POST'])
def parse_server():
  data = request.data
  data = json.loads(data.decode())
  img_s = data.get('img', None)
  front = data.get('front', None)
  res = {}
  if img_s is None:
    msg = "need img param"
    code = 400
    res['msg'] = msg
    res['code'] = code
    res['data'] = []
  elif front:
    result = CH.predict_postion(img_s)
    points = []
    for i in front:
       points.append(result.get(i))
    msg = f'已自动根据 “{front}” 排序按顺序返回汉字位置，None表示该汉字识别失败'
    code = 200
    res['msg'] = msg
    res['code'] = code
    res['data'] = points
  else:
    result = CH.predict_postion(img_s)
    res['msg'] = 'success'
    res['code'] = 200
    res['data'] = result
  return json.dumps(res)


if __name__ == '__main__':
  app.run(port=8000, host="127.0.0.1")







