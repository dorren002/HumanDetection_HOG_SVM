import cv2, os
import pickle
from sklearn import svm
import numpy as np

'''
the architecture of the folder dataset is like

-dataset
    -trainphoto
        -posphoto
        -negphoto
    -testphoto
    -testphoto_result(the sequence of the label of the testphoto)
    
all the training pictures' size is 64*128
'''

def add_label(id, feature, label):
    '''build (photoID,HOGvectors) likely pairs for training'''
    global samples
    samples[id] = np.insert(feature, 0, label)

    
def hog_label_save(path):
    '''compute the feature of the image and save them in the global variable samples, or you can put them into two variables. 
       the input, path is the directory of training sets'''
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


def svm_(train_x,train_y):
    clf = svm.SVC(kernel="linear", cache_size=800)
    clf.fit(train_x, train_y)
    return clf


if __name__ == "__main__":
    global samples
    samples = {}
    datapath = ".\\dataset\\trainphoto\\"       # your directory of training set here
    hog_label_save(datapath)
    # print("computing_hog_done")

    # print("total_train_data:", len(samples))
    train=[v for v in samples.values()]
    train = np.array(train)
    
    train_x = train[:,1:]
    train_y = train[:,0]
    # print("training...")
    clf = svm_(train_x, train_y)

    f = open('saved_model/m4.pickle', 'wb')
    pickle.dump(clf, f)
    f.close()
