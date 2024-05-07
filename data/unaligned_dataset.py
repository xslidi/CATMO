import os.path
from data.base_dataset import BaseDataset, get_transform, get_norm, _luminance, imread
from data.image_folder import make_dataset
import random
import numpy as np
import skimage
import torchvision.transforms as transforms

class UnalignedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform = get_transform(opt, luminance=opt.lum)
        self.ttensor = transforms.ToTensor()


    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        A_img = imread(A_path)
        B_img = imread(B_path)
        if self.opt.lum:
            A_img = _luminance(A_img)
            B_img = _luminance(B_img)
        elif self.opt.hsv:
            A_img = skimage.color.rgb2hsv(A_img)
            B_img = skimage.color.rgb2hsv(B_img)
            B_img = B_img.astype(np.float32)

        A_img = get_transform(self.opt, norm=False)(A_img)
        A_origin = self.ttensor(A_img)
 
        A_img_norm = get_norm(A_img, data_norm=self.opt.data_norm)
        A = self.ttensor(A_img_norm)

        B = self.transform(B_img)

        A_origin = self.ttensor(A_img)
        if self.opt.direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        out_dict = {'A': A, 'B': B, 'A_origin': A_origin,
                'A_paths': A_path, 'B_paths': B_path}
            
        return out_dict

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'

