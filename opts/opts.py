import os
import argparse
from pathlib import Path
import json

def json_serial(obj):
    if isinstance(obj, Path):
        return str(obj)

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
                            help="Pose Estimation's Body Model pretrained weights file path.")
        parser.add_argument('--pe_hand_model_path',
                            default=Path('models/pose/hand_pose_model.pth'),
                            type=Path,
                            help="Pose Estimation's Hand Model pretrained weights file path.")

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