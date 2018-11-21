import numpy as np
import os
import ntpath
import time
from util import util
import visdom


class Visualizer():
    def __init__(self,opt):
        self.display_id = opt.display_id
        #self.html = opt.isTrain and not opt.no_html

        if self.display_id >0:
            import visdom
            self.vis = visdom.Visdom(server=opt.display_server,port=opt.display_port,env=opt.display_env,raise_exceptions=True)
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)
    def throw_visdom_connection_error(self):
        print('\n\nCould not connect to Visdom server (https://github.com/facebookresearch/visdom) for displaying training progress.\nYou can suppress connection to Visdom using the option --display_id -1. To install visdom, run \n$ pip install visdom\n, and start the server by \n$ python -m visdom.server.\n\n')
        exit(1)   
    # def plot_current_losses(self,epoch,opt,losses,legend):
    #     if not hasattr(self,'plot_data'):
    #         self.plot_data ={'X':[],'Y':[],'legend':legend}
    #     self.plot_data['X'].append(epoch)
    #     self.plot_data['Y'].append(losses)
    #     try:
    #         self.vis.line(
    #             #X=np.stack([np.array(self.plot_data['X'])],1),
    #             X=np.array(self.plot_data['X']),
    #             Y=np.array(self.plot_data['Y']),
    #             opts={
    #                 'title': opt.name + 'loss over time',
    #                 'legend': self.plot_data['legend'],
    #                 'xlabel':'epoch',
    #                 'ylabel':'loss'},
    #             win = self.display_id)
    def plot_current_losses(self,epoch,opt,losses,legend):
        if not hasattr(self,'plot_data'):
            self.plot_data ={'X':[],'Y':[],'legend':legend}
        self.plot_data['X'].append(epoch)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': opt.name + 'loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel':'epoch',
                    'ylabel':'loss'},
                win = self.display_id)
                

        except ConnectionError:
            self.throw_visdom_connection_error()