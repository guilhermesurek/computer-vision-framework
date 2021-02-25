# webcam.py
import cv2
import time

import ipdb
from object_detection.object_detection import ObjDet
from pose_estimation.pose_estimation import Pose
from activity_recognition.activity_recognition import ActRec
from distance import get_shoulder_dist_from_pe, get_obj_obj_dist

class TimeMeter():
    ''' Handle time measurements. There are two modes available "time" and "fps".
    '''
    def __init__(self, display_time=2, topic=None, verbose=False):
        self.topic = topic
        self.verbose = verbose
        self.display_time = display_time
        self.reset()
    
    def reset(self):
        ''' Reset time counting.'''
        self.start_time = time.time()
        self.avg = 0
        self.sum = 0
        self.temp_count = 0
        self.acc_count = 0
        self.time = 0
        self.fps = 0
    
    def start(self):
        self.start_time = time.time()

    def count(self):
        ''' Evaluated time lapse betweed calculations.
        '''
        # update time
        self.time = time.time() - self.start_time
        # save acc sum of time elapsed
        self.sum = self.sum + self.time
        # temp count 
        self.temp_count+=1
        # acc count
        self.acc_count+=1
        # avg update
        self.avg = self.sum / self.acc_count
        # verbosity
        if self.verbose and self.topic != None:
            print(f"{self.topic} | cur time: {self.time:.2f} | avg time: {self.avg}")
        # if it's display time, calculate fps
        if self.time > self.display_time:
            # calculate fps
            self.fps = round(self.temp_count / self.time, ndigits=2)
            # verbosity
            if self.verbose:
                print(f"FPS: {self.fps}")
            # clear frame count
            self.temp_count = 0
            # restart time counting
            self.start()

def flush_var_in_frame(frame, flush_var):
    # Get frame dimensions
    (w, h, c) = frame.shape
    # Initialize positions
    x, y = [10, 10]
    # Define font size
    font_size = 25
    # Define x slide
    x_slide = 120
    # Loop over variables
    for key in flush_var:
        # Check if the key is a list or a value
        if not isinstance(flush_var[key], list):
            # Calculate postion
            y+=font_size
            # Flush info on frame
            frame = cv2.putText(frame, f"{key}: {flush_var[key]}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1)
            # If got in the end of vertical axis
            if (y + font_size >= h):
                # slide in x axis
                x+=x_slide
        else:
            # Text with key point pre defined
            value, x1, y1, cx, cy = flush_var[key]
            # Check points (x, y)
            y1 = h if y1 > h else y1
            x1 = w if x1 > w else x1
            y1 = 0 if y1 < 0 else y1
            x1 = 0 if x1 < 0 else x1
            # Flush info on frame
            frame = cv2.putText(frame, f"{key}: {value}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            frame = cv2.circle(frame, (cx, cy), 1, (0, 255, 255)) # comment
    return frame

def main_cam(opt):
    ''' Run main script with webcam.
    '''
    # Create a dict to save flush variables
    flush_var = {}

    # Instanciate the video capture object cap
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("We cannot open webcam.")
    
    # Initialize fps calculator
    fps_meter = TimeMeter(verbose=opt.verbose)

    if opt.od:
        # Instanciate the Object Detector Model
        OD_obj = ObjDet(opt)
        # Initialize detection timing
        OD_meter = TimeMeter(topic='Object Detection', verbose=opt.verbose)
    
    if opt.pe:
        # Instanciate the Pose Estimation Model
        PE_obj = Pose(opt)
        # Initialize detection timing
        PE_meter = TimeMeter(topic='Pose Estimation', verbose=opt.verbose)

    if opt.ar:
        # Instanciate the Pose Estimation Model
        AR_obj = ActRec(opt)
        # Initialize detection timing
        AR_meter = TimeMeter(topic='Activity Recognition', verbose=opt.verbose)

    # Main loop to flush webcam image
    while True:
        # Read from webcam
        _, frame = cap.read()
        # Save frame in another variable to keep original frame
        img = frame.copy()

        # AR Evaluate
        if opt.ar:
            # Save img in batch, because activity recognition needs 16 consecutives frames to execute
            AR_obj.save_in_clip(img)

        # Evaluate Models every X frames
        if fps_meter.acc_count % opt.webcam_calc_x_frames == 0:

            # AR Evaluate
            if opt.ar:
                # Start time count for PE
                AR_meter.start()
                # Apply Detection
                act_scr, act_cls = AR_obj.do_detect()
                # Save class for flush in image
                if act_cls:
                    flush_var['action']=act_cls
                # Do counting for PE
                AR_meter.count()

            # OD Evaluate
            if opt.od:
                # Start time count for OD
                OD_meter.start()
                # Resize image
                img = cv2.resize(img, (opt.od_in_w, opt.od_in_h), interpolation=cv2.INTER_AREA)
                # Apply Detection
                OD_obj.do_detect(img)
                # Print boxes and labels in the image
                img = OD_obj.plot_boxes_cv2(img, OD_obj.boxes[0])
                # Do counting for OD
                OD_meter.count()
            
            # PE Evaluate
            if opt.pe:
                # Start time count for PE
                PE_meter.start()
                # Apply Detection
                img = PE_obj.do_detect(img)
                # Do counting for PE
                PE_meter.count()
                # Calculate shoulder distance
                distances = get_shoulder_dist_from_pe(candidate=PE_obj.body_candidate, subset=PE_obj.body_subset)
                for i in range(len(distances)):
                    flush_var[f"p{i+1}"] = distances[i]

        #import ipdb; ipdb.set_trace()
        if opt.pe and opt.od:
            i=0
            for objectd in OD_obj.distances:
                if len(distances)>0 and objectd.label == 'cup':
                    flush_var[f"p_obj{i+1}"] = get_obj_obj_dist(obj1_pos=[distances[0][3], distances[0][4]],
                                                obj2_pos=[objectd.centerx, objectd.centery],
                                                img_pos=[int(img.shape[1]/2), int(img.shape[0]/2)],
                                                obj1_dist=distances[0][0],
                                                obj2_dist=objectd.dist,
                                                factor=objectd.factor)
                    i+=1


        # Calculate FPS
        fps_meter.count()
        flush_var['fps']=fps_meter.fps
        
        # Flush variables in frame
        img = flush_var_in_frame(img, flush_var)
        # Clean Flush Dict
        flush_var = {"action": flush_var['action']} if 'action' in flush_var.keys() else {}

        # Keep flushing OD result
        if opt.od:
            # Print boxes and labels in the image
            img = OD_obj.plot_boxes_cv2(img, OD_obj.boxes[0])

        # Show frame with flushes
        cv2.imshow("Web cam input", img)

        # Stop key 'q' to exit webcam input
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
    
    # Release webcam and close all cv2 windows
    cap.release()
    cv2.destroyAllWindows()