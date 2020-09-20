# webcam.py
import cv2
import time
from object_detection.object_detection import ObjDet

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
    font_size = 20
    # Define x slide
    x_slide = 120
    # Loop over variables
    for key in flush_var:
        # Calculate postion
        y+=font_size
        # Flush info on frame
        frame = cv2.putText(frame, f"{key}: {flush_var[key]}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 1)
        # If got in the end of vertical axis
        if (y + font_size >= h):
            # slide in x axis
            x+=x_slide
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

    # Main loop to flush webcam image
    while True:
        # Read from webcam
        _, frame = cap.read()
        # Save frame in another variable to keep original frame
        img = frame

        # OD Evaluate
        if opt.od:
            # Resize image
            img = cv2.resize(img, (opt.od_in_w, opt.od_in_h), interpolation=cv2.INTER_AREA)
            # Apply Detection
            OD_obj.do_detect(img)
            # Print boxes and labels in the image
            img = OD_obj.plot_boxes_cv2(img, OD_obj.boxes[0])

        # Calculate FPS
        fps_meter.count()
        flush_var['fps']=fps_meter.fps
        
        # Flush variables in frame
        img = flush_var_in_frame(img, flush_var)

        # Show frame with flushes
        cv2.imshow("Web cam input", img)

        # Stop key 'q' to exit webcam input
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
    
    # Release webcam and close all cv2 windows
    cap.release()
    cv2.destroyAllWindows()