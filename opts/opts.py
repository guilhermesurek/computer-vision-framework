import os
import argparse
from pathlib import Path
import json

def json_serial(obj):
    if isinstance(obj, Path):
        return str(obj)

def get_mean_std(value_scale, dataset):
    assert dataset in ['activitynet', 'kinetics', '0.5']

    if dataset == 'activitynet':
        mean = [0.4477, 0.4209, 0.3906]
        std = [0.2767, 0.2695, 0.2714]
    elif dataset == 'kinetics':
        mean = [0.4345, 0.4051, 0.3775]
        std = [0.2768, 0.2713, 0.2737]
    elif dataset == '0.5':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    mean = [x * value_scale for x in mean]
    std = [x * value_scale for x in std]

    return mean, std

class Options():
    ''' Class to handle all arguments and options manipulations.
    '''

    def parse_opts(self):
        ''' Define parse options, variable types, default and help info.
        '''

        # Initialize parser
        parser = argparse.ArgumentParser()

        # Define possible arguments
        ### GENERAL SECTION
        parser.add_argument('--root_path',
                            default=None,
                            type=Path,
                            help='Root directory path.')
        parser.add_argument('--verbose',
                            action='store_true',
                            help='If you want to see prints of what is happening in the code.')
        parser.add_argument('--cuda',
                            default=True,
                            type=bool,
                            help='If you want to use cuda, if available.')
        parser.add_argument('--lang',
                            default="PT",
                            type=str,
                            help='Languagem of prints and labels, if available.')
        ### OPTIONS SECTION
        parser.add_argument('--opts_src_path',
                            default=None,
                            type=Path,
                            help='Opts input file path. If not informed assume that there is no opts file available and it will assume arguments and default options.')
        parser.add_argument('--opts_dst_path',
                            default=None,
                            type=Path,
                            help='Opts output file path. If not informed and opts_output.json in the folder opts we be created.')
        ### WEBCAM SECTION
        parser.add_argument('--webcam',
                            action='store_true',
                            help='Run scripts with webcam.')
        parser.add_argument('--webcam_calc_x_frames',
                            default=16,
                            type=int,
                            help='Run models with webcam input every X frames.')
        ### OBJECT DETECTIION SECTION
        parser.add_argument('--od',
                            action='store_true',
                            help="Evaluate the Object Detection Algorithm.")
        parser.add_argument('--od_model',
                            default='Yolov4',
                            type=str,
                            help="Model type of Object Detection Algorithm. For now only support 'Yolov4'.")
        parser.add_argument('--od_n_classes',
                            default=80,
                            type=int,
                            help="Number of classes of the Object Detection Algorithm. Default 80 (Coco).")
        parser.add_argument('--od_in_h',
                            default=416,
                            type=int,
                            help="Input height for the Object Detection Model.")
        parser.add_argument('--od_in_w',
                            default=416,
                            type=int,
                            help="Input width for the Object Detection Model.")
        parser.add_argument('--od_weights_path',
                            default=Path('models/obj_det/yolov4.pth'),
                            type=Path,
                            help='Object Detection model pretrained weights file path.')
        parser.add_argument('--od_class_names_path',
                            default=Path('models/obj_det/coco.names'),
                            type=Path,
                            help='Object Detection model class names file path, if your pretrained weights use different class names.')                
        parser.add_argument('--od_conf_threshold',
                            default=0.4,
                            type=float,
                            help="Object Detection Model Confidence to filter outputs.")
        parser.add_argument('--od_nms_threshold',
                            default=0.6,
                            type=float,
                            help="Object Detection Model NSM Threshold.")
        ### POSE ESTIMATION SECTION
        parser.add_argument('--pe',
                            action='store_true',
                            help="Evaluate the Pose Estimation Algorithm.")
        parser.add_argument('--pe_body',
                            default=True,
                            type=bool,
                            help="Use body model in the evaluation of Pose Estimation Algorithm.")
        parser.add_argument('--pe_body_draw',
                            default=True,
                            type=bool,
                            help="Draw body points in the input image.")
        parser.add_argument('--pe_hand',
                            default=False,
                            type=bool,
                            help="Use hand model in the evaluation of Pose Estimation Algorithm.")
        parser.add_argument('--pe_hand_draw',
                            default=True,
                            type=bool,
                            help="Draw hand points in the input image.")
        parser.add_argument('--pe_body_model_path',
                            default=Path('models/pose/body_pose_model.pth'),
                            type=Path,
                            help="Pose Estimation Body Model pretrained weights file path.")
        parser.add_argument('--pe_body_keys_path',
                            default=Path('models/pose/body_key_points.json'),
                            type=Path,
                            help="Pose Estimation Body keys meaning file path (json).")
        parser.add_argument('--pe_hand_model_path',
                            default=Path('models/pose/hand_pose_model.pth'),
                            type=Path,
                            help="Pose Estimation Hand Model pretrained weights file path.")
        ### ACTIVITY RECOGNITION SECTION
        parser.add_argument('--ar',
                            action='store_true',
                            help="Evaluate the Activity Recognition Algorithm.")
        parser.add_argument('--ar_model_path',
                            default=Path('models/act_rec/act_rec_model.pth'),
                            type=Path,
                            help="Activity Recognition: Model's file path.")
        parser.add_argument('--ar_class_names_path',
                            default=Path('models/act_rec/hmdb51_PT.names'),
                            type=Path,
                            help="Activity Recognition: Model class names file path.")
        parser.add_argument('--ar_model',
                            default='resnet',
                            type=str,
                            help="Activity Recognition: Model's type. Default is resnet.")
        parser.add_argument('--ar_model_depth',
                            default=18,
                            type=int,
                            help="Activity Recognition: Model's depth. Default is 18.")
        parser.add_argument('--ar_threshold',
                            default=0.5,
                            type=int,
                            help="Activity Recognition: Model's threshold. Default is 0.5.")
        parser.add_argument('--ar_n_classes',
                            default=51,
                            type=int,
                            help="Activity Recognition: Model's number of classes. Default is 51 from HMDB51 dataset classes.")
        parser.add_argument('--ar_n_input_channels',
                            default=3,
                            type=int,
                            help="Activity Recognition: Model's number of input channels. Default RGB, 3 channels.")
        parser.add_argument('--ar_resnet_shortcut',
                            default='B',
                            type=str,
                            help='Activity Recognition: Shortcut type of resnet (A | B).')
        parser.add_argument('--ar_conv1_t_size',
                            default=7,
                            type=int,
                            help='Activity Recognition: Kernel size in t dim of conv1.')
        parser.add_argument('--ar_conv1_t_stride',
                            default=1,
                            type=int,
                            help='Activity Recognition: Stride in t dim of conv1.')                            
        parser.add_argument('--ar_no_max_pool',
                            action='store_true',
                            help='Activity Recognition: If true, the max pooling after conv1 is removed.')
        parser.add_argument('--ar_resnet_widen_factor',
                            default=1.0,
                            type=float,
                            help='Activity Recognition: The number of feature maps of resnet is multiplied by this value.')
        parser.add_argument('--ar_wide_resnet_k',
                            default=2,
                            type=int,
                            help='Activity Recognition: Wide resnet k.')
        parser.add_argument('--ar_resnext_cardinality',
                            default=32,
                            type=int,
                            help='Activity Recognition: ResNeXt cardinality.')                                                 
        parser.add_argument('--ar_mean_dataset',
                            default='kinetics',
                            type=str,
                            help='Activity Recognition: Dataset for mean values of mean subtraction (activitynet | kinetics | 0.5).')
        parser.add_argument('--ar_no_mean_norm',
                            action='store_true',
                            help='Activity Recognition: If true, inputs are not normalized by mean.')
        parser.add_argument('--ar_no_std_norm',
                            action='store_true',
                            help='Activity Recognition: If true, inputs are not normalized by standard deviation.')
        parser.add_argument('--ar_value_scale',
                            default=1,
                            type=int,
                            help='Activity Recognition: If 1, range of inputs is [0-1]. If 255, range of inputs is [0-255].')
        parser.add_argument('--ar_sample_size',
                            default=112,
                            type=int,
                            help='Activity Recognition: Height and width of inputs.')
        parser.add_argument('--ar_crop',
                            default='center',
                            type=str,
                            help=('Activity Recognition: Cropping method in inference. (center | nocrop)'
                                'When nocrop, fully convolutional inference is performed,'
                                'and mini-batch consists of clips of one video.'))

        # Apply parser to the inputed arguments
        args = parser.parse_args()

        # Return arguments parsed in a dictionary
        return args
    
    def load_opts_from_file(self, opt):
        ''' Load options from an options json file.
        '''
        if opt.opts_src_path is not None:

            # read json opts file data 
            with opt.opts_src_path.open('r') as opt_file:
                data = json.load(opt_file)

            # replace default info by file info
            for key in data.keys():
                opt.__dict__[key] = data[key]

        # return replaced opts dict
        return opt

    def save_opts_to_file(self, opt):
        ''' Save options to an output file, for future reference.
        '''
        # save json opts into a file
        with opt.opts_dst_path.open('w') as opt_file:
            json.dump(vars(opt), opt_file, default=json_serial)

    def set_default_opts(self, opt):
        ''' Set some default not informed options.
        '''
        # Get root path from 
        if opt.root_path is None:
            opt.root_path = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        if opt.opts_dst_path is None:
            opt.opts_dst_path = opt.root_path / 'opts/opts_output.json'
        else:
            if not os.path.isabs(opt.opts_dst_path):
                opt.opts_dst_path = opt.root_path / opt.opts_dst_path
        if opt.opts_src_path is not None:
            if not os.path.isabs(opt.opts_src_path):
                opt.opts_src_path = opt.root_path / opt.opts_src_path
        if opt.od_weights_path is not None:
            if not os.path.isabs(opt.od_weights_path):
                opt.od_weights_path = opt.root_path / opt.od_weights_path
        if opt.od_class_names_path is not None:
            if not os.path.isabs(opt.od_class_names_path):
                opt.od_class_names_path = opt.root_path / opt.od_class_names_path
        if opt.pe_body_model_path is not None:
            if not os.path.isabs(opt.pe_body_model_path):
                opt.pe_body_model_path = opt.root_path / opt.pe_body_model_path
        if opt.pe_hand_model_path is not None:
            if not os.path.isabs(opt.pe_hand_model_path):
                opt.pe_hand_model_path = opt.root_path / opt.pe_hand_model_path
        if opt.ar_class_names_path is not None:
            if not os.path.isabs(opt.ar_class_names_path):
                opt.ar_class_names_path = opt.root_path / opt.ar_class_names_path
        if opt.ar_model_path is not None:
            if not os.path.isabs(opt.ar_model_path):
                opt.ar_model_path = opt.root_path / opt.ar_model_path        
        
        opt.ar_mean, opt.ar_std = get_mean_std(opt.ar_value_scale, dataset=opt.ar_mean_dataset)

        # return result opts
        return opt
    
    @property
    def opts(self):
        # Get arguments parsed and default options
        opt = self.parse_opts()

        # Set other default options
        opt = self.set_default_opts(opt)

        # Read opts from file
        opt = self.load_opts_from_file(opt)
        
        # Write opts to file
        self.save_opts_to_file(opt)

        # return opts dict
        return opt