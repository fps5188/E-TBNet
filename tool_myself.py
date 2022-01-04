import numpy
import numpy as np
import torch
from torch import Tensor

def genConfusionMatrix(numClass, imgPredict, imgLabel):  #
    mask = (imgLabel >= 0) & (imgLabel < numClass)
    label = numClass * imgLabel[mask] + imgPredict[mask]
    count = np.bincount(label, minlength=numClass ** 2)
    confusionMatrix = count.reshape(numClass, numClass)
    # print(confusionMatrix)
    return confusionMatrix


