# Spider_Captcha
#### About Captcha:
	网易易盾、极验等主流验证码破解。
	已更新极验文字点选、极验滑动、极验语序、极验九宫格、易盾文字、易盾图标、易盾滑动、易盾乱序还原拼图、税务红黄蓝验证码、梦幻西游8位汉字识别、多种字符验证码。
#### 其验证码类型图片已经放到captcha目录下!
#### Warning！
请保证本项目仅用于研究学习！感谢配合支持！

---

#### About javascript:
To learn about basic! learn it with us! QQ群:1126310403
#### About updated:
#### 关于解决类似极验文字点选验证码的两套框架!
##### 一、目标检测框架
##### 二、文字识别框架

#### 易盾文字点选验证码模型训练思路!
##### 一、label_image 标注文字位置约400左右
##### 二、pytorch + yolov3 训练位置预测模型
##### 三、根据预测位置裁剪汉字图片
##### 四、pytorch + CNN 深层神经网络识别汉字
##### 总结：大部分的验证码用目标检测+卷积神经网络是都能搞定的！如果类似易盾图标点选就可以减少数据集，使用孪生网络进行预测！

#### About share:
验证码的识别的关键影响因素跟数据集的质量和数量影响很大！所以我也希望大家能共同一起分享数据集！
##### 如果觉得我分享的质量觉着满意，麻烦给个star! ^_^ thank you!也是每个创作者继续分享的动力！fighting!
  
#### About install:
##### 很多github上都有讲Linux系统的，本文以windows为例!
  - [x] pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
  - [x] pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
  - [x] pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
  - [x] 本套识别系统环境搭建十分简单！如果使用cuda的高级玩家，把pytorch的版本换下就ok
  - [x] [pytorch环境安装链接](https://pytorch.org/)
  - [x] 安装完毕！启动captcha_server.py 就ok!调用实例在api_requests.py里面！
  - [x] 由于方便大家下载模型和数据集！已经上传至QQ群：1126310403 群文件！
  

#### About contact:
#### 欢迎技术交流：`923185571@qq.com`
#### QQ群：1126310403
#### QQ：923185571
#### 商务合作VX：`spider_captcha`

#### 或者扫一扫加我微信，请备注GitHub，谢谢！。
<p align="center">
	<img src="./vx.JPG" alt="Sample"  width="160" height="250">
</p>

