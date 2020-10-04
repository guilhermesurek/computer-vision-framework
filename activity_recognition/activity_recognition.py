import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

from activity_recognition.models import resnet, resnet2p1d, pre_act_resnet, wide_resnet, resnext, densenet
from activity_recognition.spatial_transforms import Compose, ToTensor, Normalize, ScaleValue, Resize, Scale, CenterCrop, get_normalize_method

class ActivityRecognitionError(Exception):
    pass

class ActRec():
    ''' Activity Recognition Class to handle model, loads, predictions, etc.
    '''
    def __init__(self, opt):
        self.opt = opt
        self.thresh = opt.ar_threshold
        self.class_names = self.get_class_names()
        self.model = self.get_model()
        self.spatial_transform = self.get_spatial_transform()
        self.clip = []
    
    def get_model(self):
        # Verify model option
        assert self.opt.ar_model in [
            'resnet', 'resnet2p1d', 'preresnet', 'wideresnet', 'resnext', 'densenet'
        ]
        # Select Model
        if self.opt.ar_model == 'resnet':
            model = resnet.generate_model(
                model_depth=self.opt.ar_model_depth,
                n_classes=self.opt.ar_n_classes,
                n_input_channels=self.opt.ar_n_input_channels,
                shortcut_type=self.opt.ar_resnet_shortcut,
                conv1_t_size=self.opt.ar_conv1_t_size,
                conv1_t_stride=self.opt.ar_conv1_t_stride,
                no_max_pool=self.opt.ar_no_max_pool,
                widen_factor=self.opt.ar_resnet_widen_factor)
        elif self.opt.ar_model == 'resnet2p1d':
            model = resnet2p1d.generate_model(
                model_depth=self.opt.ar_model_depth,
                n_classes=self.opt.ar_n_classes,
                n_input_channels=self.opt.ar_n_input_channels,
                shortcut_type=self.opt.ar_resnet_shortcut,
                conv1_t_size=self.opt.ar_conv1_t_size,
                conv1_t_stride=self.opt.ar_conv1_t_stride,
                no_max_pool=self.opt.ar_no_max_pool,
                widen_factor=self.opt.ar_resnet_widen_factor)
        elif self.opt.ar_model == 'wideresnet':
            model = wide_resnet.generate_model(
                model_depth=self.opt.ar_model_depth,
                k=self.opt.ar_wide_resnet_k,
                n_classes=self.opt.ar_n_classes,
                n_input_channels=self.opt.ar_n_input_channels,
                shortcut_type=self.opt.ar_resnet_shortcut,
                conv1_t_size=self.opt.ar_conv1_t_size,
                conv1_t_stride=self.opt.ar_conv1_t_stride,
                no_max_pool=self.opt.ar_no_max_pool)
        elif self.opt.ar_model == 'resnext':
            model = resnext.generate_model(
                model_depth=self.opt.ar_model_depth,
                cardinality=self.opt.ar_resnext_cardinality,
                n_classes=self.opt.ar_n_classes,
                n_input_channels=self.opt.ar_n_input_channels,
                shortcut_type=self.opt.ar_resnet_shortcut,
                conv1_t_size=self.opt.ar_conv1_t_size,
                conv1_t_stride=self.opt.ar_conv1_t_stride,
                no_max_pool=self.opt.ar_no_max_pool)
        elif self.opt.ar_model == 'preresnet':
            model = pre_act_resnet.generate_model(
                model_depth=self.opt.ar_model_depth,
                n_classes=self.opt.ar_n_classes,
                n_input_channels=self.opt.ar_n_input_channels,
                shortcut_type=self.opt.ar_resnet_shortcut,
                conv1_t_size=self.opt.ar_conv1_t_size,
                conv1_t_stride=self.opt.ar_conv1_t_stride,
                no_max_pool=self.opt.ar_no_max_pool)
        elif self.opt.ar_model == 'densenet':
            model = densenet.generate_model(
                model_depth=self.opt.ar_model_depth,
                n_classes=self.opt.ar_n_classes,
                n_input_channels=self.opt.ar_n_input_channels,
                conv1_t_size=self.opt.ar_conv1_t_size,
                conv1_t_stride=self.opt.ar_conv1_t_stride,
                no_max_pool=self.opt.ar_no_max_pool)

        # Load pretrained model
        if self.opt.verbose:
            print('Activity Recognition: Loading pretrained model.')
        pretrain = torch.load(self.opt.ar_model_path, map_location='cpu')
        model.load_state_dict(pretrain['state_dict'])

        # If cuda, move model to cuda
        if self.opt.cuda:
            if torch.cuda.device_count() > 0 and torch.cuda.is_available():
                model = nn.DataParallel(model, device_ids=None).cuda()
        return model
    
    def get_class_names(self):
        # Check if class names file was informed
        if self.opt.ar_class_names_path == None:
            raise ActivityRecognitionError("An class names file path must be informed for the Activity Recognition Model. Use --ar_class_names_path.")
        # Load class names from file
        class_names = []
        with open(self.opt.ar_class_names_path, 'r') as fp:
            lines = fp.readlines()
        for line in lines:
            line = line.rstrip()
            class_names.append(line)
        # return class names
        return class_names

    def get_spatial_transform(self):
        normalize = get_normalize_method(self.opt.ar_mean, self.opt.ar_std, self.opt.ar_no_mean_norm,
                                     self.opt.ar_no_std_norm)
        spatial_transform = [Resize(self.opt.ar_sample_size)]
        if self.opt.ar_crop == 'center':
            spatial_transform.append(CenterCrop(self.opt.ar_sample_size))
        spatial_transform.append(ToTensor())
        spatial_transform.extend([ScaleValue(self.opt.ar_value_scale), normalize])
        spatial_transform = Compose(spatial_transform)
        return spatial_transform

    def preprocessing(self, clip):
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(Image.fromarray(np.uint8(img)).convert('RGB')) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        clip = torch.stack((clip,), 0)
        return clip

    def do_detect(self):
        # Check clip list length
        if len(self.clip)==16:
            self.model.eval()
            clip = self.preprocessing(self.clip)
            with torch.no_grad():
                outputs = self.model(clip)
                outputs = F.softmax(outputs, dim=1).cpu()
                score, class_prediction = torch.max(outputs, 1)
            return score[0], self.class_names[class_prediction[0]]
        return None, None
    
    def save_in_clip(self, img):
        ''' Function to save image to the clip list.
        '''
        # Save image in clip list
        self.clip.append(img)
        # check if has more than necessary quantity of clips and remove first
        if len(self.clip)>16:
            del self.clip[0]