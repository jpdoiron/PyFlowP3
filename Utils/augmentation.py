import os
import random
import uuid

import cv2
import numpy as np
from imgaug import augmenters as iaa

from Utils.data_aug.data_aug import RandomHorizontalFlip, RandomScale, RandomShear, RandomRotate, RandomTranslate

#seq = iaa.Sometimes(0.5, iaa.SomeOf((0, None), [
seq = iaa.SomeOf((1, 3), [
    iaa.GaussianBlur(sigma=(0.0, 1.0)),
    iaa.AdditiveGaussianNoise(scale=10),
    iaa.Sharpen(alpha=0.1),
    iaa.Sequential([
        iaa.ChangeColorspace(from_colorspace="BGR", to_colorspace="HSV"),
        iaa.WithChannels(1, iaa.Add((-50, 50))),
        iaa.WithChannels(2, iaa.Add((-100, 100))),
        iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="BGR")
    ]),
    iaa.EdgeDetect(alpha=(0.2, 0.2)),
    iaa.AddElementwise((-40, 40)),
    iaa.CoarseDropout(0.02, size_percent=0.05),
    iaa.ContrastNormalization((0.5, 1.5)),
    iaa.PiecewiseAffine(scale=(0.01, 0.02))

])



def Translate(frm, img_):
    img_, bboxes_ = RandomTranslate(0.2, diff=True)(img_, frm)
    return bboxes_, img_


def Shear(frm, img_):
    img_, bboxes_ = RandomShear(0.1)(img_, frm)
    return bboxes_, img_


def Rotate(frm, img_):
    img_, bboxes_ = RandomRotate(10)(img_, frm)
    return bboxes_, img_


def Scale(frm, img_):
    img_, bboxes_ = RandomScale(0.2, diff=True)(img_, frm)
    return bboxes_, img_


def Flip(frm, image):
    img_, bboxes_ = RandomHorizontalFlip(1)(image, frm)
    return bboxes_, img_

def PixelAugment(frm,image):
    image = seq.augment_image(image)
    return frm,image

def Rien(frm,image):
    return frm, image

my_list = [Rien,PixelAugment,Flip, Scale, Rotate,Shear,Translate]


def augment_image(image, frame,debugdir=""):
    frm = np.array([frame]).astype(np.float64)

    bboxes_, img_ = random.choice(my_list)(frm,image)
    size = max(image.shape) + 1


    if(len(bboxes_) == 0 ):
        return image, frame

    frm = bboxes_[0].astype(np.int)

    if(frm.shape != frame.shape):
        print(frm.shape,frame.shape)

    if np.sqrt(np.prod(frm[2:4] - frm[:2])) < 20:
        print("Too small")
        return image, frame

    if np.array(frm).min() < 0:
        print("plus petit que zero")
        return image, frame

    if np.array(frm).max() > size:
        print("plus grand que ", size)
        return image, frame

    if len(debugdir)>0:
        try:
            left, top, right, bottom = np.asarray(frm)
            image2 = np.ascontiguousarray(img_, dtype=np.uint8)
            image3 = cv2.rectangle(image2, tuple((left, top)), tuple((right, bottom)), (0, 0, 255), 3)
            unique_filename = str(uuid.uuid4())
            file = os.path.join(debugdir, "{}.jpg".format(unique_filename))
            cv2.imwrite(file,image3)
        except:
            pass
        finally:
            pass

    return img_, frm



# -----------------------------------------------------------------------#

# Helper funtions for data augumentation for training the network #
def coord_translate(box, tr_x, tr_y):
    coords = np.array(box)
    coords[0::2] = coords[0::2] + tr_x
    coords[1::2] = coords[1::2] + tr_y
    coords = coords.astype(np.int64)
    coords = coords.tolist()
    return coords


def coord_scale(box, sc):
    coords = np.array(box)
    coords = coords * sc
    coords = coords.astype(np.int64)
    coords = coords.tolist()
    return coords

def testAugment():
    from skimage import data
    # For simplicity, we use the same image here many times
    astronaut = data.astronaut()
    import matplotlib.pyplot as plt
    plt.imshow(astronaut)
    plt.show()
    frame = (100,5,300,200)
    import time

    start = time.time()

    for x in range(100):
        #a, f = augment_image(astronaut, frame,"/tmp")
        a, f = augment_image(astronaut, frame)

        #show_box(a, f)

    end = time.time()
    print(end - start)


if __name__ == '__main__':
    testAugment()

