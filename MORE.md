## 第一步：数据集预处理

IRNIA数据集有正例2416幅，负例1218幅，其中正例为已裁剪为64*128尺寸的图像，负例在进行裁减后共12180幅。从上述图像各抽取20%作为验证集记录其分类结果分开存储，剩余图像为训练集。

## 第二步：提取图像HOG特征

提取图像的特征可依赖opencv包或参照HOG.py文件，对应算法见参考文献1，提取结束后将提取的特征和对应标签暂存。【train->hog_label_save()】

## 第三步：训练

x-HOG特征向量

y-label（1-有行人，0-没有行人）

## 第四步：验证

将验证集投入训练得到的模型，比对结果。



对于多尺度图像，可以使用opencv-hogdetecter对象的多尺度检测实现多尺度检测，后续合并重合的方框可以得到不重合的框。

## 参考文献：
[1]Dalal, N. and Triggs, B., “Histograms of oriented gradients for human detection,” in [Computer Vision and Pattern Recognition, 2005. CVPR 2005. IEEE Computer Society Conference on ], 1, 886–893 vol. 1 (June).
[2]利用Hog特征和SVM分类器进行行人检测: https://blog.csdn.net/carson2005/article/details/7841443
[3]行人检测数据集转换代码：https://download.csdn.net/download/fan0920/10935382
