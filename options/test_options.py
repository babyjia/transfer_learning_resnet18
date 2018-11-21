from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        print("TestOptions_intizlize")

        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--niter', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        
        self.isTrain = False
        return parser