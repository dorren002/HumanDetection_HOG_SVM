import pickle,cv2,os,json
import PIL.Image as Image
import numpy as np


'''
测试数据集目录结构：
-testphoto
    -tst000001.jpg
    -tst000002.jpg
    ...
    
结果保存json格式
{‘filename’：result}
'''


# 测试数据检测
def eval(path,model):
    global result
    hog = cv2.HOGDescriptor()
    for p,dirs,fnames in os.walk(path):
        for fname in fnames:
            pth = os.path.join(p, fname)
            img=cv2.imread(pth)
            feature = hog.compute(img)
            if feature is None:
                continue
            else:
                feature = feature.ravel().reshape(1,3780)
                res=model.predict(feature)
                result[fname[3:-4]]=int(res[0])


# 任意图像行人检测
def test(image):
    result = {}
    hog = cv2.HOGDescriptor()
    h,w = image.shape
    # 63*128滑窗检测，跳数为8
    for x in range(0,h-128,4):
        for y in range(0,w-64,8):
            block = image[x:x+129,y:y+65]
            feature = hog.compute(block)
            if feature is None:
                continue
            else:
                feature = feature.ravel().reshape(1,3780)
                res = model.predict(feature)
                if int(res) == 1:
                    result[(x,y)]=1
    return result


# 尺度变换，输入尺度坐标和点坐标
def scalerchange(scale, dots):
    res={}
    for k in dots.keys():
        x,y = k
        x1, y1 = 1/2**scale*x, 1/2**scale*y
        res[(x1,y1)] = 1
    return res


# 直接绘制标记框，输入为检测图像和检测结果
def drawVirtrualBox(image,res):
    for v in res.keys():
        x,y = v
        cv2.rectangle(image, (y,x), (y+64,x+128), (184,1,2) ,1)
    cv2.imshow("result", image)
    cv2.waitKey()


# 合并重叠标记框并绘制，输入为检测图像和检测结果
def drawVirtrualBox2(image, res):
    for k,v in res.items():
        if v==0:
            continue
        x,y = k
        maxy, maxx, minx, miny = y + 64, x + 128, x, y
        for k1 in res.keys():
            x1,y1 = k1
            if minx<=x1<=maxx and miny<=y1<=maxy or miny<=y1+64<=maxy:
                res[k1]=0
                maxy = max(maxy, y1+64)
                maxx = max(maxx, x1+128)
                miny = min(miny, y1)
                minx = min(minx, x1)
        cv2.rectangle(image, (miny,minx), (maxy,maxx), (255, 1, 2), 2)
    cv2.imshow("result", image)
    cv2.waitKey()


# 测试集评估，输入为测试集路径
def testset(testset):
    eval(testset, model)
    with open(".\\results\\myresult_m3.json", 'w', encoding='utf-8')as json_file:
        json.dump(result, json_file, ensure_ascii=False)


if __name__=="__main__":
    # 加载模型
    f = open('saved_model/m4.pickle','rb')
    model = pickle.load(f)
    f.close()
    result={}
    testphoto = "dataset/testphoto"
    # testset(testphoto)
    img = np.asarray(Image.open("test21.jpg").convert('L'))
    result = test(img)
    drawVirtrualBox(cv2.imread("test21.jpg"),result)
    drawVirtrualBox2(cv2.imread("test21.jpg"),result)


