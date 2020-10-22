
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import random
from keras.preprocessing.image import save_img
from skimage.transform import resize
# from scipy.misc import imsave, imresize depreciated
from scipy.optimize import fmin_l_bfgs_b   # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import warnings

random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

CONTENT_IMG_PATH = "content.jpg"
STYLE_IMG_PATH = "style.jpg"


CONTENT_IMG_H = 500
CONTENT_IMG_W = 500

STYLE_IMG_H = 500
STYLE_IMG_W = 500

CONTENT_WEIGHT = 0.7    # Alpha weight.
STYLE_WEIGHT = 0.3      # Beta weight.
TOTAL_WEIGHT = 1.0

TRANSFER_ROUNDS = 3



#=============================<Helper Fuctions>=================================
'''
TODO: implement this.
This function should take the tensor and re-convert it to an image.
'''
def deprocessImage(img):
    img = img.reshape((CONTENT_IMG_H, CONTENT_IMG_W, 3))
    img = img[:, :, ::-1]
    return img


def gramMatrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram



#========================<Loss Function Builder Functions>======================

def styleLoss(style, gen):
    # Correct for numfilters?
    return K.sum(K.square(gramMatrix(style) - gramMatrix(gen))) / (4.0 * (TRANSFER_ROUNDS^2) * ((STYLE_IMG_H * STYLE_IMG_W)^2))


def contentLoss(content, gen):
    return K.sum(K.square(gen - content))


def totalLoss(x):
    return CONTENT_WEIGHT * contentLoss(x) + STYLE_WEIGHT(x) * styleLoss(x)





#=========================<Pipeline Functions>==================================

def getRawData():
    print("   Loading images.")
    print("      Content image URL:  \"%s\"." % CONTENT_IMG_PATH)
    print("      Style image URL:    \"%s\"." % STYLE_IMG_PATH)
    cImg = load_img(CONTENT_IMG_PATH)
    tImg = cImg.copy()
    sImg = load_img(STYLE_IMG_PATH)
    print("      Images have been loaded.")
    return ((cImg, CONTENT_IMG_H, CONTENT_IMG_W), (sImg, STYLE_IMG_H, STYLE_IMG_W), (tImg, CONTENT_IMG_H, CONTENT_IMG_W))



def preprocessData(raw):
    img, ih, iw = raw
    img = img_to_array(img)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = resize(img, (ih, iw, 3))
    img = img.astype("float64")
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


'''
TODO: Allot of stuff needs to be implemented in this function.
First, make sure the model is set up properly.
Then construct the loss function (from content and style loss).
Gradient functions will also need to be created, or you can use K.Gradients().
Finally, do the style transfer with gradient descent.
Save the newly generated and deprocessed images.
'''
def styleTransfer(cData, sData, tData):
    print("   Building transfer model.")
    contentTensor = K.variable(cData)
    styleTensor = K.variable(sData)
    genTensor = K.placeholder((1, CONTENT_IMG_H, CONTENT_IMG_W, 3))
    inputTensor = K.concatenate([contentTensor, styleTensor, genTensor], axis=0)
    model = vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=inputTensor)
    outputDict = dict([(layer.name, layer.output) for layer in model.layers])
    print("   VGG19 model loaded.")
    loss = 0.0
    styleLayerNames = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
    contentLayerName = "block5_conv2"
    print("   Calculating content loss.")
    contentLayer = outputDict[contentLayerName]
    contentOutput = contentLayer[0, :, :, :]
    genOutput = contentLayer[2, :, :, :]
    cl = contentLoss(contentOutput, genOutput)
    sl = 0.0
    print("   Calculating style loss.")
    for layerName in styleLayerNames:
        styleLayer = outputDict[layerName]
        styleOutput = styleLayer[1, :, :, :]
        genOutput = styleLayer[2, :, :, :]
        sl += styleLoss(styleOutput, genOutput)
    loss = CONTENT_WEIGHT * cl + STYLE_WEIGHT * sl

    gradient = K.gradients(loss, genTensor)
    outputs = [loss]
    outputs += gradient
    kFunction = K.function([genTensor], outputs)

    class Wrapper:

        def loss(self, x):
            x = x.reshape((1, CONTENT_IMG_H, CONTENT_IMG_W, 3))
            outs = kFunction([x])
            self._gradients = outs[1].flatten().astype("float64")
            return outs[0]

        def gradients(self, x):
            return self._gradients

    wrapper = Wrapper()

    print("   Beginning transfer.")
    for i in range(TRANSFER_ROUNDS):
        print("   Step %d." % i)
        position, tLoss, dictionary = fmin_l_bfgs_b(wrapper.loss, tData.flatten(), fprime=wrapper.gradients, maxfun=1000)
        print("      Loss: %f." % tLoss)
        img = deprocessImage(position)
        saveFile = "generatedImg" + str(i) + ".jpg"
        save_img(saveFile, img)   #Uncomment when everything is working right.
        print("      Image saved to \"%s\"." % saveFile)
    print("   Transfer complete.")





#=========================<Main>================================================

def main():
    print("Starting style transfer program.")
    tf.compat.v1.disable_eager_execution()
    raw = getRawData()
    cData = preprocessData(raw[0])   # Content image.
    sData = preprocessData(raw[1])  # Style image.
    tData = preprocessData(raw[2])   # Transfer image.
    styleTransfer(cData, sData, tData)
    print("Done. Goodbye.")



if __name__ == "__main__":
    main()
