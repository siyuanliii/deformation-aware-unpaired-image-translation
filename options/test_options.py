from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        #  Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=1500, help='how many test images to run')
        parser.add_argument('--epoch_GS', type=str, default='latest', help='which model to use for deformation')
        parser.add_argument('--epoch_GI', type=str, default='latest', help='which model to use for image transfer')
        parser.add_argument('--source_anno_dir', type=str, default='./dataset/unpaired_dataset/fly/syn_anno.pth', help='which model to use for image transfer')
        parser.add_argument('--transfer_anno', action='store_true', help='if specified, transfer the annotation')
        parser.add_argument('--render_pose',action='store_true', help='if specified, render the pose')  
        parser.add_argument('--evaluate',action='store_true', help='if specified, evaluate the performance') 
        parser.add_argument('--test_anno_dir', type=str, default='./dataset/test/fly/test_anno.pth', help='test_anno_dir')
        parser.add_argument('--pck_range', type=int, default=45, help='threshold range of pck')
        
        parser.set_defaults(model='test')
        # To avoid cropping, the loadSize should be the same as fineSize
        parser.set_defaults(loadSize=parser.get_default('fineSize'))
        self.isTrain = False
        return parser
