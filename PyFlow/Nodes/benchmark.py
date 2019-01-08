import os
import cv2
import numpy as np

from PyFlow.Core import Node
from PyFlow.Core.AGraphCommon import DataTypes

from Utils.image_utils import draw_detection_on_image, transform_box
from Utils.yolo1.utils import detect_image


def iou(bboxes1, bboxes2):
    x11, y11, x12, y12 = bboxes1
    x21, y21, x22, y22 = bboxes2
    xA = np.maximum(x11, x21)
    yA = np.maximum(y11, y21)
    xB = np.minimum(x12, x22)
    yB = np.minimum(y12, y22)

    if xB <= xA or yB <= yA:
        return 0.0

    inter_area = (xA - xB) * (yA - yB)
    b1_area = (x12 - x11) * (y12 - y11)
    b2_area = (x22 - x21) * (y22 - y21)
    iou = inter_area / float(b1_area + b2_area - inter_area)

    return iou


def log_results(model, found, time_average, total_iou, total_boxes, labels, log_path):
    directory = os.path.join(log_path, "benchmark")
    if not os.path.exists(directory):
        os.makedirs(directory)

    file = os.path.join(directory, "benchmark_results.txt")
    file = open(file, "w")

    line_list = []
    line_list.append("obj_class | \ttotal | \tfound\n")
    for label_class in labels:
        count, found = labels[label_class]
        #TODO get the real name for the classes
        label_line = "{} \t{} \t{}\n".format(label_class, found, count)
        line_list.append(label_line)

    if found == 0:
        benchmark_output = "rien trouvé"
    else:
        if len(total_iou) > 0:
            benchmark_output = "found:{} total:{} found_ratio:{:4f} iou_avg:{:4f} temps:{:4f}".format(len(total_iou),
                                                                                                      total_boxes, (
                                                                                                              len(
                                                                                                                  total_iou) / total_boxes * 100),
                                                                                                      np.average(
                                                                                                          total_iou),
                                                                                                      np.average(
                                                                                                          time_average))
        else:
            benchmark_output = "found:{} total:{} found_ratio:{:4f} iou_avg:{:4f} temps:{:4f}".format(len(total_iou),
                                                                                                      total_boxes, (
                                                                                                              len(
                                                                                                                  total_iou) / total_boxes * 100),
                                                                                                      -1,
                                                                                                      np.average(
                                                                                                          time_average))
    line_list.append(benchmark_output)

    for line in line_list:
        print(line)
    file.writelines(line_list)
    file.close()


def log_image(model, image, image_name, box, log_path):
    directory = os.path.join(log_path, "benchmark")
    if not os.path.exists(directory):
        os.makedirs(directory)

    image = draw_detection_on_image(model, image, [box])
    file = os.path.join(directory, "{}.png".format(image_name))
    cv2.imwrite(file, image)


class benchmark(Node):
    def __init__(self, name, graph):
        super(benchmark, self).__init__(name, graph)
        self.in0 = self.addInputPin('In', DataTypes.Exec, self.compute)
        self.completed_pin = self.addOutputPin('Completed', DataTypes.Exec)

        self.input_model_pin = self.addInputPin('input_model', DataTypes.Any, self.compute, defaultValue=None)
        self.image_location_pin = self.addInputPin('image_location', DataTypes.String, self.compute, defaultValue="")
        self.annotation_file_pin = self.addInputPin('annotation_file', DataTypes.Files, self.compute, defaultValue="Annotation.txt")
        self.class_offset_pin = self.addInputPin('class_offset', DataTypes.Int, self.compute, defaultValue=0)
        self.log_folder_pin = self.addInputPin('log_folder', DataTypes.String, self.compute, defaultValue="Logs")
        self.enable_debug_pin = self.addInputPin('enable_debug', DataTypes.Bool, self.compute, defaultValue=False)

    @staticmethod
    def pinTypeHints():
        '''
            used by nodebox to suggest supported pins
            when drop wire from pin into empty space
        '''
        return {'inputs': [], 'outputs': []}

    @staticmethod
    def category():
        '''
            used by nodebox to place in tree
            to make nested one - use '|' like this ( 'CatName|SubCatName' )
        '''
        return 'Testing|function'

    @staticmethod
    def keywords():
        '''
            used by nodebox filter while typing
        '''
        return ['becnmark']

    @staticmethod
    def description():
        '''
            used by property view and node box widgets
        '''
        return 'benchmark model'

    # TODO should be more generic than detect image. For the moment it will do but it should work with more than images
    def compute(self):
        try:
            debug_benchmark = self.enable_debug_pin.getData()

            data_dir = self.image_location_pin.getData()
            class_offset = self.class_offset_pin.getData()
            annotation = self.annotation_file_pin.getData()
            model = self.input_model_pin.getData()
            log_folder = self.log_folder_pin.getData()

            with open(annotation) as f:
                lines = f.readlines()
            values = set(map(lambda x: x.split()[0], lines))

            newlist = dict()
            for x in lines:
                key, box = x.split()
                if key in newlist:
                    newlist[key].append(box)
                else:
                    newlist[key] = [box]

            totalIOU = []
            timeAvr = []
            totalbox = 0
            label = {}
            for idx, val in enumerate(values):
                if (idx % 100) == 0:
                    print("Processing : {} on {} ".format(idx, len(values)))
                filename = os.path.join(data_dir, val)
                image = cv2.imread(filename)

                # Debug code
                display_image = False
                reference_displayed = False
                # Debug end
                for box in newlist[val]:
                    totalbox += 1
                    x1, y1, x2, y2, cl = list(map(int, box.split(',')))
                    out_classes, out_boxes, out_scores, transform, time, detection_image = detect_image(image, model, (
                        (x1 + x2) // 2, (y1 + y2) // 2))
                    timeAvr.append(time)

                    # ironman
                    # cl = cl - 1
                    # mattelcar
                    # cl = cl-19
                    cl = cl - class_offset
                    if cl not in label:
                        label[cl] = [0, 0]

                    cnt, found = label[cl]
                    cnt += 1
                    if cl in out_classes:
                        found += 1

                        if not isinstance(out_classes, list):
                            idx_cl = out_classes.tolist().index(cl)
                        else:
                            idx_cl = out_classes.index(cl)
                        b1 = x1, y1, x2, y2
                        boxPred = out_boxes[idx_cl]

                        b2 = boxPred
                        iou = iou(b1, b2)

                        # Debug code
                        if debug_benchmark:
                            if iou < 0.5:
                                display_image = True
                                print("Classe: ", model.class_names[cl])

                                if not reference_displayed:
                                    image = draw_detection_on_image(model, image, out_boxes, out_classes, out_scores,
                                                                    (255, 0, 0))
                                    reference_displayed = True

                                image = draw_detection_on_image(model, image, [b1], [], [], (0, 255, 0), thickness=10)
                        # Debug end
                        totalIOU.append(iou)
                    else:
                        if debug_benchmark:
                            #fixer le classname properly
                            log_image(model, detection_image,
                                           "{}_{}".format(os.path.splitext(val)[0], cl),
                                           transform_box([x1, y1, x2, y2], transform), log_folder)

                    label[cl] = cnt, found

                # Debug code
                if debug_benchmark and display_image:
                    import matplotlib.pyplot as plt
                    plt.imshow(image)
                    plt.show(block=True)
                # Debug end

            log_results(model, found, timeAvr, totalIOU, totalbox, label, log_folder)

        except Exception as e:
            import traceback
            import sys
            traceback.print_exception(type(e), e, sys.exc_info()[2], limit=1, file=sys.stdout)
