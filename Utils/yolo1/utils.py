import numpy as np
import cv2
from scipy.special import expit as sigmoid
from timeit import default_timer as timer

from imgaug import augmenters as iaa

from Utils.image_utils import image_resize2, transform_box

seq = iaa.Sometimes(0.5, iaa.SomeOf((0, None), [
    iaa.GaussianBlur(sigma=(0.0, 1.0)),
    iaa.AdditiveGaussianNoise(scale=10),
    iaa.Sharpen(alpha=0.1),
    iaa.Sequential([
        iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
        iaa.WithChannels(1, iaa.Add((-50, 50))),
        iaa.WithChannels(2, iaa.Add((-100, 100))),
        iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")
    ]),
    iaa.EdgeDetect(alpha=(0.2, 0.2)),
    iaa.AddElementwise((-40, 40)),
    iaa.CoarseDropout(0.02, size_percent=0.05),
    iaa.ContrastNormalization((0.5, 1.5)),
    iaa.PiecewiseAffine(scale=(0.01, 0.02))

]))


def draw_boxes(img, bboxes_w_conf, color=(0, 0, 255), thick=2, draw_dot=False, radius=7):
    # Make a copy of the image
    draw_img = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes_w_conf:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(draw_img, tuple(bbox[0]), tuple(bbox[1]), color, thick)
        cv2.putText(draw_img, '{:.2f}'.format(bbox[2]), tuple(bbox[0]), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,0),2)
        if draw_dot:
            centre = (np.array(bbox[0]) + np.array(bbox[1])) // 2
            cv2.circle(draw_img, tuple(centre), radius=radius, color=(0, 255, 0), thickness=-1)
    # Return the image copy with boxes drawn
    return draw_img

def get_boxes(nn_output, cutoff=0.2,label=False):
    '''
    Extracts boxes from the network prediction with greater confidence score that 'cutoff'
    # Arguments
    nn_output: numpy array of shape (1573,)
    cutoff: confidence score cutoff
    dims: dimensions to scale the output to. useful for images that are not the
            same dimensions as the images the network is trained on
    '''
    WIDTH_NORM = 224
    HEIGHT_NORM = 224
    GRID_NUM = 11
    X_SPAN = WIDTH_NORM/GRID_NUM
    Y_SPAN = HEIGHT_NORM/GRID_NUM
    X_NORM = WIDTH_NORM/GRID_NUM
    Y_NORM = HEIGHT_NORM/GRID_NUM
    obj_class= nn_output[:363].reshape(11, 11, 3)
    conf_scores = nn_output[363:363+242].reshape(11,11,2)
    if(label):
        xywh = nn_output[-968:].reshape(11,11,2,4)
    else:
        xywh = sigmoid(nn_output[-968:].reshape(11, 11, 2, 4))


    indx_max_ax2 = np.argmax(conf_scores, axis=2)
    # indx_max_ax2 looks like:
    # array([[0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1],
    #    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    #    [0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1],
    #    .
    #    .
    #    .
    #    [1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0]], dtype=int64)
    i, j = np.meshgrid(np.arange(11), np.arange(11), indexing='ij')
    indx_max = np.stack((i,j,indx_max_ax2), axis=2)
    # array([[[ 0,  0,  0],
    #     [ 0,  1,  0],
    #     [ 0,  2,  0],
    #     .
    #     .
    #     [ 0, 10,  1]],
    #
    #    [[ 1,  0,  0],
    #     [ 1,  1,  0],
    #     .
    #     .
    #     [10,  8,  1],
    #     [10,  9,  1],
    #     [10, 10,  0]]], dtype=int64)
    indx_max = indx_max.reshape(-1,3)
    winning_bbox_conf_score = conf_scores[indx_max[:,0], indx_max[:,1], indx_max[:,2]].reshape(11,11)
    indx_cutoff = np.argwhere(winning_bbox_conf_score >= cutoff)

    last_indx = indx_max_ax2[indx_cutoff[:,0], indx_cutoff[:,1]]
    last_indx = np.expand_dims(last_indx, axis=1)

    detection_indx = np.concatenate((indx_cutoff, last_indx), axis=1)

    # xywh_detection = xywh[detection_indx[:,0], detection_indx[:,1], detection_indx[:,2], :]
    # #print(xywh_detection)
    # xywh_detection[:,0] = xywh_detection[:,0] * X_NORM
    # xywh_detection[:,1] = xywh_detection[:,1] * Y_NORM
    #
    # xywh_detection[:,2] = xywh_detection[:,2] * WIDTH_NORM
    # xywh_detection[:,3] = xywh_detection[:,3] * HEIGHT_NORM

    bboxes = []
    for a, b, c in zip(detection_indx[:,0], detection_indx[:,1], detection_indx[:,2]):
        x = (xywh[a,b,c,0] * X_NORM + b * X_SPAN)
        y = (xywh[a,b,c,1] * Y_NORM + a * Y_SPAN)
        w = (xywh[a,b,c,2] * WIDTH_NORM)
        h = (xywh[a,b,c,3] * HEIGHT_NORM)

        x1, x2 = int(x-w/2), int(x+w/2)
        y1, y2 = int(y-h/2), int(y+h/2)

        bboxes.append(((x1,y1), (x2,y2), conf_scores[a,b,c], np.argmax(obj_class[a,b])))

    return bboxes


def nonmax_suppression(bboxes, iou_cutoff = 0.05):
    '''
    Suppress any overlapping boxes with IOU greater than 'iou_cutoff', keeping only
    the one with highest confidence scores
    # Arguments
    bboxes: array of ((x1,y1), (x2,y2)), c) where c is the confidence score
    iou_cutoff: any IOU greater than this is considered for suppression
    '''
    suppress_list = []
    max_list = []
    for i in range(len(bboxes)):
        box1 = bboxes[i]
        for j in range(i+1, len(bboxes)):
            box2 = bboxes[j]
            iou = iou_value(box1[:2], box2[:2])
            #print(i, " & ", j, "IOU: ", iou)
            if iou >= iou_cutoff:
                if box1[2] > box2[2]:
                    suppress_list.append(j)
                else:
                    suppress_list.append(i)
                    continue
    #print('suppress_list: ', suppress_list)
    for i in range(len(bboxes)):
        if i in suppress_list:
            continue
        else:
            max_list.append(bboxes[i])
    return max_list


def iou_value(box1, box2):
    '''
    calculate the IOU of two given boxes
    '''
    (x11, y11) , (x12, y12) = box1
    (x21, y21) , (x22, y22) = box2

    x1 = max(x11, x21)
    x2 = min(x12, x22)
    w = max(0, (x2-x1))

    y1 = max(y11, y21)
    y2 = min(y12, y22)
    h = max(0, (y2-y1))

    area_intersection = w*h
    area_combined = abs((x12-x11)*(y12-y11) + (x22-x21)*(y22-y21) + 1e-3)

    return area_intersection/area_combined


def detect_image(image_in, model, center=(0, 0)):
    image, transform = image_resize2(image_in,(224,224), center=center)

    image = ProcessInput(image)

    image_data = np.expand_dims(image, axis=0)  # Add batch dimension.


    start = timer()
    pred = model.predict(image_data)
    end = timer()
    time = (end - start)

    bboxes1 = get_boxes(pred[0], cutoff=0.3)
    bboxes = nonmax_suppression(bboxes1, iou_cutoff=0.05)

    out_boxes = []
    out_scores = []
    out_classes = []

    for bbox in bboxes:
        (x, y), (x1, y1), conf, cl = bbox
        box = transform_box((x,y,x1,y1),transform, inverse=True)
        out_boxes.append(box)
        out_scores.append(conf)
        out_classes.append(cl)

    return out_classes, out_boxes, out_scores, transform, time, image


def ProcessInput(image):
    from tensorflow.python.keras.applications.imagenet_utils import preprocess_input
    '''image channel transformation for training / inferencing'''
    # b, g, r = cv2.split(image)  # get b,g,r
    # image = cv2.merge([r, g, b])  # switch it to rgb
    image = preprocess_input(image, mode='tf')
    return image