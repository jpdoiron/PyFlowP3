import math
import random
from enum import Enum

import cv2
import matplotlib.pyplot as plt
import numpy as np


class ResizeMode(Enum):
    CROP_ONLY = 2
    SQUASH = 3
    PAD = 4


def image_resizeWWW(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)

        dim = (math.ceil(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(math.ceil(h * r)))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized, dim


def image_resize2(image, dim, center=(0,0), mode=ResizeMode.CROP_ONLY):
    xcenter, ycenter = center
    width, height = dim

    if width is None or height is None:
        return image

    xtrans = ytrans = 0
    xratio = yratio =1

    old_size = image.shape[:2]  # old_size is in (height, width) format

    if mode == ResizeMode.CROP_ONLY:
        min_size = min(old_size)
        if xcenter > 0:
            xtrans = clamp(xcenter - min_size // 2, 0, old_size[1] - min_size)
        if ycenter > 0:
            ytrans = clamp(ycenter - min_size // 2, 0, old_size[0] - min_size)

        desired_size = min(width, height)
        xratio = yratio = float(desired_size) / min(old_size)

    elif mode == ResizeMode.PAD:
        desired_size = min(width, height)
        xratio = yratio = float(desired_size) / max(old_size)

    elif mode == ResizeMode.SQUASH:
        xratio = width / old_size[1]
        yratio = height / old_size[0]

    src_pts = np.float32([[0, 0], [0, old_size[1] * yratio], [old_size[0] * xratio, old_size[1] * yratio]])
    dst_pts = np.float32([[xtrans, ytrans], [xtrans, ytrans + old_size[1]], [xtrans + old_size[0], ytrans + old_size[1]]])
    transform_out = cv2.getAffineTransform(dst_pts, src_pts)

    image = cv2.warpAffine(image, transform_out, dim)
    return image, np.array(transform_out)

def inverse_transform(transform):
    as_square = [transform[0], transform[1], np.array([0, 0, 1])]
    inversed = np.linalg.inv(as_square)

    return inversed[0:2]


def transform_boxes(boxes, transform, maxSize=(1e10,1e10), inverse = False):
    b_out=[]
    for box in boxes:
        b_out.append(transform_box(box, transform, maxSize))

    return b_out

def transform_box(box, transform, maxSize=(1e10,1e10), inverse = False):
    work_transform = transform
    if inverse:
        work_transform = inverse_transform(np.copy(transform))

    x, y, x1, y1 = box[:4]
    dst = cv2.transform(np.array([[[x, y]], [[x1, y1]]]), work_transform)
    out_box = np.array(dst).flatten()

    out_box = ClampBox(out_box, maxSize)

    return out_box


def ClampBox(box, maxSize):
    t = np.tile(maxSize, 2)
    out_box = np.clip(box, 0, t)
    return out_box


def draw_detection_on_image(model, image, out_boxes, out_classes=[], out_scores=[], box_color=(0, 0, 255), label_color=(255, 255, 0), thickness=3):
    index = 0
    for curr_box in out_boxes:
        left, top, right, bottom = np.asarray(curr_box).astype(np.int)
        image = cv2.rectangle(image, tuple((left, top)), tuple((right, bottom)), box_color, thickness)
        if len(out_classes) > 0 and len(out_scores) > 0:
            label = '{} {:.2f}'.format(model.class_names[out_classes[index]], out_scores[index])
            image = cv2.putText(image, label, tuple((left, top)), cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 2)
        index = index + 1
    return image

def show_box(image, box, obj_class="", score=0):
    show_boxes(image, [box], obj_class, score)

def show_boxes(image, boxes, obj_class="", score=0):

    r = lambda: random.randint(0, 255)

    for box in boxes:
        left, top, right, bottom = np.asarray(box).astype(np.int)
        image = cv2.rectangle(image, tuple((left, top)), tuple((right, bottom)), (r(),r(),r()), 3)
        if len(obj_class):
            label = '{} {:.2f}'.format(obj_class, score)
            image = cv2.putText(image, label, tuple((left, top)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    plt.imshow(image)
    plt.show()


def clamp(n, smallest, largest): return max(smallest, min(n, largest))

#Test Section
def test_resizeImage(image_path, size, mode, window_caption):
    image = cv2.imread(image_path)
    image2, trans = image_resize2(image, size, mode=mode)

    f = plt.figure()
    f.add_subplot(1, 2, 1,  title="original")
    plt.imshow(image[..., ::-1])
    f.add_subplot(1, 2, 2, title=window_caption)
    plt.imshow(image2[..., ::-1])
    plt.show(block=True)

def test_all_image_resize():
    #crop
    test_resizeImage("images/img1.png", (416, 416), ResizeMode.CROP_ONLY, "Image 1 Crop no center")
    test_resizeImage("images/img1.png", (416, 416), ResizeMode.CROP_ONLY, "Image 1 Crop x trans")
    test_resizeImage("images/img1.png", (416, 416), ResizeMode.CROP_ONLY, "Image 1 Crop xy trans")

    test_resizeImage("images/img2.png", (416, 416), ResizeMode.CROP_ONLY, "Image 2 Crop no center")
    test_resizeImage("images/img3.png", (416, 416), ResizeMode.CROP_ONLY, "Image 2 Crop no center")

    # squash
    test_resizeImage("images/img1.png", (416, 416), ResizeMode.SQUASH, "Image 1 squash no center")
    test_resizeImage("images/img1.png", (416, 416), ResizeMode.SQUASH, "Image 1 squash x trans")
    test_resizeImage("images/img1.png", (416, 416), ResizeMode.SQUASH, "Image 1 squash xy trans")

    test_resizeImage("images/img2.png", (416, 416), ResizeMode.SQUASH, "Image 2 squash no center")
    test_resizeImage("images/img3.png", (416, 416), ResizeMode.SQUASH, "Image 2 squash no center")

    # PAD
    test_resizeImage("images/img1.png", (416, 416), ResizeMode.PAD, "Image 1 pad no center")
    test_resizeImage("images/img1.png", (416, 416), ResizeMode.PAD, "Image 1 pad x trans")
    test_resizeImage("images/img1.png", (416, 416), ResizeMode.PAD, "Image 1 pad xy trans")

    test_resizeImage("images/img2.png", (416, 416), ResizeMode.PAD, "Image 2 pad no center")
    test_resizeImage("images/img3.png", (416, 416), ResizeMode.PAD, "Image 2 pad no center")


if __name__ == '__main__':
    test_all_image_resize()