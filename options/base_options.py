import argparse
import torch
from util import util
import os

class BaseOptions():
    def __init__(self):
        self.initialized = False
    
    
    def initialize(self, parser):

        parser.add_argument('--dataroot', default='dataset/plate/', required=True, help='path of dataset')
        parser.add_argument('--loadSize', type=int, default=256,help='scale image to this size')
        parser.add_argument('--fineSize', type=int, default=224,help='the crop to this size')
        parser.add_argument('--name', required=True, help='the experiment of experiment')
        parser.add_argument('--display_server', type=str, default='http://10.25.0.244')
        parser.add_argument('--display_id',type=int,default=1,help='0,not use 1 for use visdom')
        parser.add_argument('--display_env',type=str,default='main',help='visdom displat environment name')
        parser.add_argument('--display_port', type=int, default=8088, help='the port of visdom')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--gpu_ids', type=str, default='0,1,2,3', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--model', type=str, required=True, default='none', help='densenet201,pnasnet5lar,resnet101,my_senet154')
        parser.add_argument('--class_weight', default = True, help='using class weight')
        self.initialized = True
        print('baseoptions_intitialize')
        return parser
    
    
    def gater_options(self):

        if not self.initialized:
            print("not initialized")
            parser = argparse.ArgumentParser(
                formatter_class = argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        
        #opt, _ = parser.parse_known_args()

        self.parser = parser

        return parser.parse_args()









    def print_options(self,opt):
        message = ''
        message += '-------------------Options--------------------\n'
        for key,value in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(key)
            if value != default:
                comment = '\t[default:%s]'% str(default)
            message += '{:<24}:{:>34}{}\n'.format(str(key),str(value),str(comment))

        message +='-------------------Options--------------------\n'

        print(message)


        #save to the disk
        save_dir = os.path.join(opt.checkpoints_dir,opt.name)
        util.mkdirs(save_dir)
        file_name = os.path.join(save_dir,'opt.txt')
        with open(file_name,'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')



    def parse(self):

        opt = self.gater_options()
        opt.isTrain = self.isTrain


        self.print_options(opt)


        self.opt = opt
        return opt