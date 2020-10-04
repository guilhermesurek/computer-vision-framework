# object_detection.py
from object_detection.od_model import Yolov4
import time
import numpy as np
import torch
import cv2
import math

class ObjectDetectionError(Exception):
    pass

class ObjDet():
    ''' Object Detection Class to handle model, loads, predictions, etc.
    '''
    def __init__(self, opt):
        self.opt = opt
        self.height = opt.od_in_h
        self.width = opt.od_in_w
        self.n_classes = opt.od_n_classes
        self.weights_path = opt.od_weights_path
        self.conf_thresh = opt.od_conf_threshold
        self.nms_thresh = opt.od_nms_threshold
        self.class_names = self.get_class_names()
        self.model = self.get_model()
    
    def get_model(self):
        # Verify model option
        assert self.opt.od_model in ['Yolov4']
        # Select Model
        if self.opt.od_model == 'Yolov4':
            model = Yolov4(yolov4conv137weight=None, n_classes=self.opt.od_n_classes, inference=True)
        # Load pretrained model
        if self.weights_path is not None:
            pretrained_dict = torch.load(self.weights_path, map_location='cpu') #torch.device('cuda'))
            model.load_state_dict(pretrained_dict)
        # If cuda, move model to cuda
        if self.opt.cuda:
            if torch.cuda.device_count() > 0 and torch.cuda.is_available():
                model.cuda()
        return model

    def get_class_names(self):
        # Check if class names file was informed
        if self.opt.od_class_names_path == None:
            raise ObjectDetectionError("An class names file path must be informed for the Object Detection Model. Use --od_class_names_path.")
        # Load class names from file
        class_names = []
        with open(self.opt.od_class_names_path, 'r') as fp:
            lines = fp.readlines()
        for line in lines:
            line = line.rstrip()
            class_names.append(line)
        # return class names
        return class_names

    def do_detect(self, img, conf_thresh=None, nms_thresh=None):
        # Set model to eval mode
        self.model.eval()
        # Start counting time
        t0 = time.time()

        # Transform image based on shape
        if type(img) == np.ndarray and len(img.shape) == 3:  # cv2 image
            img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
        elif type(img) == np.ndarray and len(img.shape) == 4:
            img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float().div(255.0)
        else:
            raise ObjectDetectionError("Unknow image type.")

        # If cuda available, load image to cuda
        if self.opt.cuda:
            if torch.cuda.device_count() > 0 and torch.cuda.is_available():
                img = img.cuda()
        # Make torch transformations
        img = torch.autograd.Variable(img)
        # Save preprocessed time
        t1 = time.time()
        # Run model
        output = self.model(img)
        # Save model inference time
        t2 = time.time()
        # If verbose, print information
        if self.opt.verbose:
            print('-----------------------------------')
            print('       OBJECT DETECTION MODEL      ')
            print('           Preprocess : %f' % (t1 - t0))
            print('      Model Inference : %f' % (t2 - t1))
            print('-----------------------------------')
        # Check parameters
        if conf_thresh == None:
            conf_thresh = self.conf_thresh
        if nms_thresh == None:
            nms_thresh = self.nms_thresh
        # Get Boxes
        self.boxes = post_processing(img, conf_thresh, nms_thresh, output, verbose=self.opt.verbose)
        return self.boxes
    
    def plot_boxes_cv2(self, img, boxes=None, savename=None, class_names=None, color=None):
        if boxes == None:
            boxes = self.boxes[0]
        if class_names == None:
            class_names = self.class_names
        img = np.copy(img)
        colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)

        def get_color(c, x, max_val):
            ratio = float(x) / max_val * 5
            i = int(math.floor(ratio))
            j = int(math.ceil(ratio))
            ratio = ratio - i
            r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
            return int(r * 255)

        width = img.shape[1]
        height = img.shape[0]
        for i in range(len(boxes)):
            box = boxes[i]
            x1 = int(box[0] * width)
            y1 = int(box[1] * height)
            x2 = int(box[2] * width)
            y2 = int(box[3] * height)

            if color:
                rgb = color
            else:
                rgb = (255, 0, 0)
            if len(box) >= 7 and class_names:
                cls_conf = box[5]
                cls_id = box[6]
                #print('%s: %f' % (class_names[cls_id], cls_conf))
                classes = len(class_names)
                offset = cls_id * 123457 % classes
                red = get_color(2, offset, classes)
                green = get_color(1, offset, classes)
                blue = get_color(0, offset, classes)
                if color is None:
                    rgb = (red, green, blue)
                img = cv2.putText(img, class_names[cls_id], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rgb, 1)
                # Distance
                #dist=47*484.04/(abs(x1-x2))
                #print(f"distance: {dist:0.3f}")
            img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 1)
        if savename:
            print("save plot results to %s" % savename)
            cv2.imwrite(savename, img)
        return img

def nms_cpu(boxes, confs, nms_thresh=0.5, min_mode=False):
    # print(boxes.shape)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]

    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]

        keep.append(idx_self)

        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        if min_mode:
            over = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]
    
    return np.array(keep)

def post_processing(img, conf_thresh, nms_thresh, output, verbose=True):

    # anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
    # num_anchors = 9
    # anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    # strides = [8, 16, 32]
    # anchor_step = len(anchors) // num_anchors

    # [batch, num, 1, 4]
    box_array = output[0]
    # [batch, num, num_classes]
    confs = output[1]

    t1 = time.time()

    if type(box_array).__name__ != 'ndarray':
        box_array = box_array.cpu().detach().numpy()
        confs = confs.cpu().detach().numpy()

    num_classes = confs.shape[2]

    # [batch, num, 4]
    box_array = box_array[:, :, 0]

    # [batch, num, num_classes] --> [batch, num]
    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)

    t2 = time.time()

    bboxes_batch = []
    for i in range(box_array.shape[0]):
       
        argwhere = max_conf[i] > conf_thresh
        l_box_array = box_array[i, argwhere, :]
        l_max_conf = max_conf[i, argwhere]
        l_max_id = max_id[i, argwhere]

        bboxes = []
        # nms for each class
        for j in range(num_classes):

            cls_argwhere = l_max_id == j
            ll_box_array = l_box_array[cls_argwhere, :]
            ll_max_conf = l_max_conf[cls_argwhere]
            ll_max_id = l_max_id[cls_argwhere]

            # Check same class box overlap
            keep = nms_cpu(ll_box_array, ll_max_conf, nms_thresh)
            
            if (keep.size > 0):
                ll_box_array = ll_box_array[keep, :]
                ll_max_conf = ll_max_conf[keep]
                ll_max_id = ll_max_id[keep]

                for k in range(ll_box_array.shape[0]):
                    bboxes.append([ll_box_array[k, 0], ll_box_array[k, 1], ll_box_array[k, 2], ll_box_array[k, 3], ll_max_conf[k], ll_max_conf[k], ll_max_id[k]])
        
        bboxes_batch.append(bboxes)

    t3 = time.time()

    if verbose:
        print('-----------------------------------')
        print('       max and argmax : %f' % (t2 - t1))
        print('                  nms : %f' % (t3 - t2))
        print('Post processing total : %f' % (t3 - t1))
        print('-----------------------------------')
    
    return bboxes_batch