import json

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
        self.body_keys_mean = self.get_body_keys()
        self.body_candidate = None
        self.body_subset = None

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
        # Body Key Points
        if self.opt.pe_body:
            self.body_candidate, self.body_subset = self.body(img)
            if self.opt.pe_body_draw:
                img = draw_bodypose(img, self.body_candidate, self.body_subset)
        # Hand Key Points
        if self.opt.pe_hand:
            self.hand_peaks = self.hand(img)
            if self.opt.pe_hand_draw:
                img = draw_handpose(img, self.hand_peaks)
        # Save key points with body part names
        self.save_key_points()
        return img
    
    def get_body_keys(self):
        ''' Function to get the body keys meaning from file.
        '''
        with open(self.opt.pe_body_keys_path) as f:
            data = json.load(f)
        return data[self.opt.lang]

    def save_key_points(self):
        ''' Function to save body key points.
        '''
        # Initialize dict
        self.body_keys = {}
        # Loop over each person found
        for p in range(len(self.body_subset)): # Subset shape: [Persons x 20] where first 18 are body key points
            self.body_keys[p] = {}
            for k in range(18):
                index = int(self.body_subset[p][k])
                if index == -1:
                    # Body key point not found
                    x, y, score = [-1, -1, -1]
                else:
                    # Body key point found
                    x, y, score = self.body_candidate[index][0:3]
                # Get body part name
                body_part = self.body_keys_mean[str(k)]
                # Save body key points
                self.body_keys[p][body_part] = {
                    "coord": [x, y],
                    "index": k,
                    "score": score
                }