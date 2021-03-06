import numpy as np
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf


class tf_encoder(object):
    def __init__(self,level):
        self.k = level
        #self.onehotencoder = OneHotEncoder(n_values=self.k, sparse=False)

    """
    input:natural image arr:n*w*h*c
    return: quantisized image n*w*h*c
    """

    def quantization(self,arr):
        quant = tf.zeros(tf.shape(arr))
        for i in range(1, self.k):
            temp = tf.cast(arr > 1.0 * i / self.k, tf.float32)
            quant = quant + temp
            #quant[arr > 1.0 * i / self.k] += 1
        return quant

        #quant = np.zeros(arr.shape)
        #for i in range(1, self.k):
        #    quant[arr > 1.0 * i / self.k] += 1
        #return quant

    """
    input:quantisized img shape:n*w*h*c
    retun:one-hot coded image shape:n*w*h*c*k
    """

    def onehot(self,arr):
        print('arr shape ', arr.get_shape())
        n, w, h = arr.get_shape()
        arr = tf.reshape(arr,[n,w*h])
        arr = tf.one_hot(arr,depth=self.k)
        arr = tf.reshape(arr, [n, w, h, self.k])
        arr = tf.transpose(arr, perm=[0,3,1,2])
        return arr


#        n, w, h = arr.shape
#        arr = arr.reshape(n, -1)
#        arr = self.onehotencoder.fit_transform(arr)
#        arr = arr.reshape(n, w, h, self.k)
#        arr = arr.transpose(0, 3, 1, 2)
#        return arr

    """
    input:one-hot coded img shape:n*w*h*c*k
    retun:trmp coded image shape:n*w*h*c*k
    """

    def tempcode(self, arr):
        tempcode = tf.zeros(tf.shape(arr))
        for i in range(self.k):
            tempcode[:,i,:,:] = tf.reduce_sum(arr[:,:i+1, :, :], axis = 1,keepdims=True)
        return tempcode


        #tempcode = np.zeros(arr.shape)
        #for i in range(self.k):
        #    tempcode[:, i, :, :] = np.sum(arr[:, :i + 1, :, :], axis=1)
        #return tempcode

    def tempencoding(self,arr):
        return self.tempcode(self.onehot(self.quantization(arr)))

    def onehotencoding(self,arr):
        return self.onehot(self.quantization(arr))


    """
    from a thermometerencoding image to a normally coded image, for some visualization usage
    """

    def temp2img(self,tempimg):
        img = np.sum(tempimg, axis=1)
        img = np.ones(img.shape) * (self.k + 1) - img
        img = img * 1.0 / self.k
        return img
