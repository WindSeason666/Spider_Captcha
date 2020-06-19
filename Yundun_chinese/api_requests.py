# 请求方式
import base64,json
import requests,os

path = r'C:\Users\spyder\Desktop\yidun_chinese\images'
for i in os.listdir(path):
    f = open(os.path.join(path,i), 'rb')
    s = base64.b64encode(f.read()).decode()
    res=requests.post(url='http://127.0.0.1:8000/captcha',data=json.dumps({'img':s}))
    data = res.json()
    print(data)
    data = data.get('data')
    f.close()
    item = {}
    for k,v in data.items():
        x = v.get('x')
        item[x] = k
    k = sorted(item.keys())
    k = ''.join([item.get(i) for i in k])
    os.rename(os.path.join(path,i),path+'\\'+k+'.jpg')



