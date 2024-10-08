from .base_model import BaseModel
from . import networks
from .cycle_gan_model import CycleGANModel
from util.util import ReviseImage
import torch

class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        assert not is_train, 'TestModel cannot be used in train mode'
        parser = CycleGANModel.modify_commandline_options(parser, is_train=False)
        parser.set_defaults(dataset_mode='single')
        parser.set_defaults(resize_or_crop='none')

        parser.add_argument('--model_suffix', type=str, default='',
                            help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will'
                            ' be loaded as the generator of TestModel')

        return parser

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = []
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B']
        if not self.opt.lum and not self.opt.hsv:
            self.visual_names.append('revise_B')
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['G' + opt.model_suffix]

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.norm_g, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, True, opt.spectral_norm)

        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see BaseModel.load_networks
        setattr(self, 'netG' + opt.model_suffix, self.netG)

    def set_input(self, input):
        # we need to use single_dataset mode
        self.real_A = input['A'].to(self.device)
        self.image_paths = input['A_paths']
        self.origin_A = input['A_origin'].to(self.device)

    def forward(self):

        if self.opt.amp:
            with torch.cuda.amp.autocast():
                self.fake_B = self.netG(self.real_A)
        else:
            self.fake_B = self.netG(self.real_A)
        if not self.opt.lum and not self.opt.hsv:
            revise_img = ReviseImage(self.fake_B, self.origin_A)
            revise_img.saved_batches()
            self.revise_B = revise_img.recover
