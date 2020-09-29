from pose_estimation.body import Body
from pose_estimation.hand import Hand
from pose_estimation.util import draw_bodypose, draw_handpose

class PoseEstimationError(Exception):
    pass

class Pose():
    ''' Pose Estimation Class to handle model, loads, predictions, etc.
    '''
    def __init__(self, opt):
        self.opt = opt
        self.body = self.get_body_model()
        self.hand = self.get_hand_model()

    def get_body_model(self):
        ''' Function to get the body model.
        '''
        if self.opt.pe_body:
            model = Body(self.opt.pe_body_model_path)
            return model
        return None
    
    def get_hand_model(self):
        ''' Function to get the hand model.
        '''
        if self.opt.pe_hand:
            model = Hand(self.opt.pe_hand_model_path)
            return model
        return None
    
    def do_detect(self, img):
        ''' Do de pose estimation.
        '''
        if self.opt.pe_body:
            self.body_candidate, self.body_subset = self.body(img)
            if self.opt.pe_body_draw:
                img = draw_bodypose(img, self.body_candidate, self.body_subset)
        if self.opt.pe_hand:
            self.hand_peaks = self.hand(img)
            if self.opt.pe_hand_draw:
                img = draw_handpose(img, self.hand_peaks)
        return img