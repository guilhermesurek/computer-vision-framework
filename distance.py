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
                centerx = int((person_pts[0][0] + person_pts[1][0])/2)
                centery = int((person_pts[0][1] + person_pts[1][1])/2)
                #distances.append([person_pts[0], person_pts[1], round(dist, ndigits=2)])
                distances.append([round(dist, ndigits=2), person_pts[0][0]-10, person_pts[0][1]-40, centerx, centery])
    return distances

def get_obj_obj_dist(obj1_pos, obj2_pos, img_pos, obj1_dist, obj2_dist, factor):
    # Factor pixels
    f_px = factor #obj1_dist/(484.04 / 0.58)
    # Calculate objs x-dist to image center
    obj1_aal = f_px * abs(obj1_pos[0] - img_pos[0])            # AA' - Projection of obj1 center in x axis
    obj2_bbl = f_px * abs(obj2_pos[0] - img_pos[0])            # BB' - Projection of obj2 center in x axis
    # Calculate in between dist
    dist_albl = abs(obj1_dist - obj2_dist)              # Eq. 6 MA = CB' - CA'
    # Calculate
    blo = obj2_bbl * dist_albl / (obj1_aal + obj2_bbl)  # Eq. 7 BM/MA = BB'/B'O
    bo = sqrt(blo**2 + obj2_bbl**2)                     # Eq. 8 BO = SQRT( B'O **2 + BB' ** 2 )
    dist_obj1_obj2 = round(bo * dist_albl / blo, 1)               # Eq. 9 AB/MA = BO/B'O
    return [dist_obj1_obj2, obj2_pos[0], obj2_pos[1], obj2_pos[0], obj2_pos[1]]

class ObjDistanceError(Exception):
    pass

class ObjDistance():
    ''' Person or Object Distance and Appearance Class to handle dimensions, distances, etc.
    '''
    def __init__(self, obj_type, label, pos, shoulder_lenght=None):
        if not isinstance(obj_type, str):
            raise ObjDistanceError(f"obj_type must be a string. Received {type(obj_type)}.")
        if not obj_type in ['object', 'person']:
            raise ObjDistanceError(f"obj_type must be 'object' or 'person'. Received {obj_type}.")
        self.type = obj_type
        if obj_type == 'object':
            self.pos = pos      # x1, y1, x2, y2
            self.height = abs(self.pos[0] - self.pos[2])
            self.width = abs(self.pos[1] - self.pos[3])
            self.centerx = int((self.pos[0] + self.pos[2])/2)
            self.centery = int((self.pos[1] + self.pos[3])/2)
            self.diagonal = euclidean((self.pos[0], self.pos[1]), (self.pos[2], self.pos[3]))
            self.label = label
            self.diagonal_real = 15.8
            self.factor = self.diagonal_real / self.diagonal
            f = 484.04 #*0.58
            self.dist= round(f*15.8/self.diagonal, 1)
        if obj_type == 'person':
            self.shoulder_lenght = shoulder_lenght
        self.dist_to_cam = None