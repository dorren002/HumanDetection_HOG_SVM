import numpy as np
import math

'''
hg = HOG()        # 新建示例
hg.compute(img)   # 提取img图像HOG特征，发挥1*3780维特征向量 
'''

class HOG():
    # 计算梯度时的算子
    win = np.array([-1, 0, 1])

    # 初始化参数为细胞单元尺寸，block尺寸和跳数stride
    def __init__(self, cell_size = 8, block_size=16, stride = 8):
        self.cell_size = cell_size
        self.block_size = block_size
        self.stride = stride

    # Gamma校正
    def GammaCorrection(self, image):
        image = image / 255.0
        res = np.power(image, 0.5)
        return res

    # 梯度计算，返回梯度模矩阵和相位矩阵
    def gradientComputation(self, src):
        GxMat, GyMat = self.filt(src)
        h,w = src.shape
        MagMat, PhaseMat = np.zeros((h, w)), np.zeros((h, w))
        for x in range(h):
            for y in range(w):
                MagMat[x][y] = np.sqrt(GxMat[x][y]**2 + GyMat[x][y]**2)
                tmp = math.degrees(math.atan(GyMat[x][y]/GxMat[x][y])) if GxMat[x][y]!=0 else 90
                PhaseMat[x][y] = tmp if tmp>0 else tmp+180
        return MagMat, PhaseMat

    # 细胞单元统计梯度方向，输入梯度模矩阵和相位矩阵
    def Binning(self, MagMat, PhaseMat):
        h, w = MagMat.shape
        sizex,sizey = int(h/self.cell_size), int(w/self.cell_size)
        bins = np.zeros((sizex,sizey,9))
        for x in range(sizex):
            for y in range(sizey):
                bins[x][y]= self.plus(MagMat[self.cell_size*x:self.cell_size*(x+1),self.cell_size*y:self.cell_size*(y+1)],PhaseMat[self.cell_size*x:self.cell_size*(x+1),self.cell_size*y:self.cell_size*(y+1)])
        return bins

    # 归一化
    def Normalizing(self,bins):
        h,w,depth = bins.shape
        res = np.zeros((0))
        for x in range(h-1):
            for y in range(w-1):
                vec = bins[x:x+2,y:y+2].flatten()
                sum_ = np.sum(vec)
                vec = vec/sum_
                res = np.append(res,vec)
        return res

    # 新建实例后通过compute方法提取hog特征
    def compute(self, src):
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        MatMat, PhaseMat = self.gradientComputation(src)
        bins = self.Binning(MatMat,PhaseMat)
        Hog = self.Normalizing(bins)
        return Hog

    # 一个细胞单元的统计
    def plus(self, MagMat, PhaseMat):
        bin = np.zeros((9))
        for i in range(self.cell_size):
            for j in range(self.cell_size):
                b = PhaseMat[i][j]/20
                if int(b)==9:
                    bin[8] = MagMat[i][j]
                    continue
                w1, w2 = b-int(b), int(b)+1-b
                bin[int(b)-1] = w1*MagMat[i][j]
                bin[int(b)] = w2*MagMat[i][j]
        # print(bin)
        return bin

    # 边缘填充
    def crop(self,src):
        h,w = src.shape
        res = np.zeros((h+2,w+2))
        res[1:-1,1:-1]=src
        res[0,1:-1]=src[0]
        res[-1,1:-1]=src[-1]
        res[1:-1,0]=src[:,0]
        res[1:-1,-1]=src[:,0]
        return res

    # 计算xy两个方向的梯度，对应返回两个矩阵
    def filt(self,src):
        h,w = src.shape
        window_croped = self.crop(src)
        GxMat, GyMat = np.zeros(src.shape), np.zeros(src.shape)
        for x in range(1,h):
            for y in range(1,w):
                GyMat[x][y] = np.sum(window_croped[x,y - 1:y + 2] * self.win)
                GxMat[x][y] = np.sum(window_croped[x - 1:x + 2,y] * self.win)
        return GxMat, GyMat