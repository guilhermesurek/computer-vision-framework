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
        ### OPTIONS SECTION
        parser.add_argument('--opts_src_path',
                            default=None,
                            type=Path,
                            help='Opts input file path. If not informed assume that there is no opts file available and it will assume arguments and default options.')
        parser.add_argument('--opts_dst_path',
                            default=None,
                            type=Path,
                            help='Opts output file path. If not informed and opts_output.json in the folder opts we be created.')
        

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