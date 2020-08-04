import cv2, os
import pickle
from sklearn import svm
import numpy as np

'''
数据集文件夹结构
-dataset
    -trainphoto
        -posphoto
        -negphoto
    -testphoto(打乱后的测试集图片）
    -testphoto_result(测试集标签用于比对)
'''


# 临时储存HOG特征和标记对  格式：(标签，HOG特征)
def add_label(id, feature, label):
    global samples
    samples[id] = np.insert(feature, 0, label)


# 获取hog特征并保存，输入数据集路径
def hog_label_save(path):
    label = 0
    i=0
    count = 0
    hog = cv2.HOGDescriptor()
    for p,dirs,fnames in os.walk(path):
        if i==2:
            label = 1
        for fname in fnames:
            pth = os.path.join(p, fname)
            img = cv2.imread(pth)
            if label == 1:
                img=cv2.resize(img,(64,128))
            feature = hog.compute(img)
            if feature is None:
                pass
            else:
                feature = feature.ravel()
                add_label(count, feature, label)
                count+=1
        i+=1


# svm训练函数
def svm_(train_x,train_y):
    clf = svm.SVC(kernel="linear", cache_size=800)
    clf.fit(train_x, train_y)
    return clf


if __name__ == "__main__":
    global samples
    samples = {}
    datapath = ".\\dataset\\trainphoto\\"
    hog_label_save(datapath)
    print("hog_done")

    print("total_train_data:", len(samples))
    train=[v for v in samples.values()]
    train = np.array(train)
    # 降临时储存的(标签-hog)拆分并放入SVM
    train_x = train[:,1:]
    train_y = train[:,0]
    print("training...")
    clf = svm_(train_x, train_y)

    f = open('saved_model/m4.pickle', 'wb')
    pickle.dump(clf, f)
    f.close()
