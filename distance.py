# distance.py
from math import sqrt
from scipy.spatial.distance import euclidean

def get_shoulder_dist_from_pe(candidate, subset):
    ''' From the pose estimation results (cadidate and subset) extract shoulder points and calculate their euclidean distance.
    '''
    distances = []
    # Check if there is any information in the vectors
    if len(candidate)>0 and len(subset)>0:
        people_sh_points = []
        # For each person body keypoints, extract shoulder points
        for sub in subset:
            sh_points = [(int(cand[0]), int(cand[1])) for cand in candidate if cand[3]==int(sub[2]) or cand[3]==int(sub[5])]
            if len(sh_points)==2:
                people_sh_points.append(sh_points)
        #points = [(int(cand[0]), int(cand[1])) for cand in candidate if cand[3]==int(subset[0,2]) or cand[3]==int(subset[0,5])]
        if len(people_sh_points) > 0:
            for person_pts in people_sh_points:
                dist = dist=43*484.04*0.58/(euclidean(person_pts[0], person_pts[1]))
                #distances.append([person_pts[0], person_pts[1], round(dist, ndigits=2)])
                distances.append([round(dist, ndigits=2), person_pts[0][0]-10, person_pts[0][1]-40])
    return distances

class ObjDistanceError(Exception):
    pass

class ObjDistance():
    ''' Person or Object Distance and Appearance Class to handle dimensions, distances, etc.
    '''
    def __init__(self, obj_type, height=None, width=None, shoulder_lenght=None):
        if not isinstance(obj_type, str):
            raise ObjDistanceError(f"obj_type must be a string. Received {type(obj_type)}.")
        if not obj_type in ['object', 'person']:
            raise ObjDistanceError(f"obj_type must be 'object' or 'person'. Received {obj_type}.")
        self.type = obj_type
        if obj_type == 'object':
            self.height = height
            self.width = width
            if self.height and self.width and (isinstance(self.height, float) or isinstance(self.height, int)) and (isinstance(self.width, float) or isinstance(self.width, int)):
                self.diagonal = sqrt(self.height**2 + self.width**2)
        if obj_type == 'person':
            self.shoulder_lenght = shoulder_lenght
        self.dist_to_cam = None