# webcam.py
import cv2
import time

class fps():
    ''' Handle fps calculations.
    '''
    def __init__(self, display_time=2, verbose=False):
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.display_time = display_time
        self.verbose = verbose
    
    def count(self):
        ''' Count frames and calculate fps. 
        '''
        # count frame
        self.frame_count+=1
        # update time
        self.time = time.time() - self.start_time
        # if it's display time, calculate fps
        if self.time > self.display_time:
            # calculate fps
            self.fps = round(self.frame_count / self.time, ndigits=2)
            # verbosity
            if self.verbose:
                print(f"FPS: {self.fps}")
            # clear frame count
            self.frame_count = 0
            # restart time counting
            self.start_time = time.time()
        # return fps value
        return self.fps

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

    # Instanciated the video capture object cap
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("We cannot open webcam.")
    
    # Initiate fps calculator
    fps_calculator = fps(verbose=opt.verbose)

    # Main loop to flush webcam image
    while True:
        # Read from webcam
        ret, frame = cap.read()

        # Calculate FPS
        flush_var['fps']=fps_calculator.count()
        
        # Flush variables in frame
        img = flush_var_in_frame(frame, flush_var)

        # Show frame with flushes
        cv2.imshow("Web cam input", img)

        # Stop key 'q' to exit webcam input
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
    
    # Release webcam and close all cv2 windows
    cap.release()
    cv2.destroyAllWindows()