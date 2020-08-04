## Histogram of Oriented Gradients for Human Detection

This is a simple implementation of Histogram of Oriented Gradients for Human Detection, and used SVM as classifier. Please star if you like this implementation.



#### Rely ons

```powershell
$ pip install scikit-learn
$ pip install opencv-python
```



#### Use

```powershell
$ python train.py # for training
$ python evaluate.py # evaluate
$ python seq2jpeg.py # transfer the seq videos to jpg images
$ python vbb2xml.py # transfer the vbb annotations to xml files
```



#### Pre-Trained Models

There are one pretrained models in saved_models folder, please use pickle to load the model.

```python
>>>import pickle 
>>>pickle.load(filename)
```



#### Dataset

1. IRNIA human Dataset(http://pascal.inrialpes.fr/data/human/)

2. Caltech Pedestrian Detection(http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/) 

   You need to use seq2jpeg.py and vbb2xml.py to preconditioning the data.

