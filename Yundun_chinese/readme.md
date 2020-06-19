
#### pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
#### pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
#### pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
#### 在data文件夹下建立三个子文件夹（Annotations、images与ImageSets，labels后续使用脚本生成）其中Annotations存放xml文件，images图像，ImageSets新建Main文件存放train与test文件（脚本生成），labels是标签文件
#### 新建两个py文件，分别放入以下两段代码，并依次运行
#### data目录下新建cat.name写上预测的类别名字，不懂的可以参考coco.name
#### cat.data，文件内容如下，配置训练的数据，同理参考coco.data
#### cfg文件修改
#### 主要修改3*3共9处，分别为最后的通道数、类别数、随机多尺度训练开关
#### python train.py --data data/cat.data --cfg cfg/yolov3-cat.cfg --weights weights/yolov3.weights --epochs 10
#### python test.py --data-cfg data/cat.data --cfg cfg/yolov3-cat.cfg --weights weights/latest.pt
#### python detect.py --data-cfg data/cat.data --cfg cfg/yolov3-cat.cfg --weights weights/best.pt
#### python -c "from utils import utils; utils.plot_results()"


