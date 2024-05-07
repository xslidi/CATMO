import torch
import torch.nn as nn
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from util.util import gram_matrix, vgg_preprocess, ReviseImage
from util.vgg import Vgg16
from packaging import version

class PatchNCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)

        # neg logit

        # Should the negatives from the other samples of a minibatch be utilized?
        # In CUT and FastCUT, we found that it's best to only include negatives
        # from the same image. Therefore, we set
        # --nce_includes_all_negatives_from_minibatch as False
        # However, for single-image translation, the minibatch consists of
        # crops from the "same" high-resolution image.
        # Therefore, we will include the negatives from the entire minibatch.
        if self.opt.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss

class CUTModel(BaseModel):
    def name(self):
        return 'CUTModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default CycleGAN did not use dropout
        # parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
            parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
            parser.add_argument('--netF_nc', type=int, default=256)
            parser.add_argument('--nce_idt', action='store_true', help='use NCE loss for identity mapping: NCE(G(Y), Y))')
            parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
            parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
            parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
            parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            action='store_true',
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # define the number of times D trained when G trained once
        self.d_iter = opt.d_iter
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['D', 'G']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['real_A', 'fake_B', 'revise_B']
        visual_names_B = ['real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]
        if self.opt.lambda_NCE > 0.0:
            self.loss_names.append('NCE')
        if self.isTrain and self.opt.nce_idt:
            visual_names_B.append('idt_B')
            self.loss_names.append('NCE_Y')
        if self.opt.style_weight > 0.0:
            self.loss_names.append('style')

        if self.opt.style_weight > 0.0:
            gpu_ids = self.opt.gpu_ids
            self.vgg = Vgg16(requires_grad=False).to(gpu_ids[0])

        self.visual_names = visual_names_A + visual_names_B
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D', 'F']
        else:  # during test time, only load Gs
            self.model_names = ['G']

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G, D
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm_g, 
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, 
                                      True, opt.spectral_norm)

        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.norm_g, not opt.no_dropout, 
                                      opt.init_type, opt.init_gain, self.gpu_ids, opt.netF_nc)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, 
                                          opt.norm_d, use_sigmoid, opt.init_type, opt.init_gain, 
                                          self.gpu_ids, opt.spectral_norm)
            self.criterionNCE = []
            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, loss=self.opt.loss).to(self.device)
            self.criterionIdt = torch.nn.L1Loss()
            self.L1loss = torch.nn.L1Loss()
            self.mseloss = torch.nn.MSELoss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr_g, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr_d, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.origin_A = input['A_origin'].to(self.device)

    def forward(self):
        self.fake_B = self.netG(self.real_A) 
        if self.opt.lum or self.opt.hsv:  
            self.revise_B = self.fake_B   
        else:
            revise_img = ReviseImage(self.fake_B, self.origin_A)
            revise_img.saved_batches()
            self.revise_B = revise_img.recover  

    def backward_D_basic(self, netD, real, fake, loss='none'):
        pred_real = netD(real)
        pred_fake = netD(fake.detach())
        # Real
        if loss == 'hinge':
            loss_D_real = nn.ReLU()(1.0 - pred_real).mean()
        elif loss == 'wasserstein':
            loss_D_real = - pred_real.mean()
        elif loss == 'relative':
            rel_t = pred_real-pred_fake
            rel_t.requires_grad_(True)
            loss_D_real = self.criterionGAN(rel_t, True)
        else:    
            loss_D_real = self.criterionGAN(pred_real, True)
        # Fake        
        if loss == 'hinge':
            loss_D_fake = nn.ReLU()(1.0 + pred_fake).mean()
        elif loss == 'wasserstein':
            loss_D_fake = pred_fake.mean()
        elif loss == 'relative':
            rel_f = pred_fake-pred_real
            rel_f.requires_grad_(True)
            loss_D_fake = self.criterionGAN(rel_f, False)
        else:
            loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        if loss == 'hinge' or loss == 'wasserstein':
            loss_D = loss_D_real + loss_D_fake
        else:
            loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward(retain_graph=True)
        return loss_D

    def style_loss(self, img, style, hdr=False, loss='l2'):
        style_loss = 0
        weight = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        if img.shape[1] == 1 and style.shape[1] == 1:
            img = img.repeat(1,3,1,1)
            style = style.repeat(1,3,1,1)
        if hdr:
            style = 2 * (style - torch.min(style)) / (torch.max(style) - torch.min(style)) - 1
        else:
            style = vgg_preprocess(style)
            img = vgg_preprocess(img)
        
        style_vgg = self.vgg(style)
        img_vgg = self.vgg(img)
        if loss == 'l2':
            loss_fn = self.mseloss
        elif loss == 'l1':
            loss_fn = self.L1loss
        for i in range(len(style_vgg)):
            if self.opt.gram_matrix:
                style_vgg[i] = gram_matrix(style_vgg[i]) 
                img_vgg[i] = gram_matrix(img_vgg[i])
            style_loss += weight[i] * loss_fn(style_vgg[i], img_vgg[i]) / len(style_vgg)
        return style_loss

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)


        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers


    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.backward_D()                  # calculate gradients for D
            self.backward_G()                  # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr_g, betas=(self.opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_F)


    def backward_D(self):
        fake_B = self.fake_B_pool.query(self.revise_B)
        self.loss_D = self.backward_D_basic(self.netD, self.real_B, fake_B, self.opt.loss)

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.idt_B = self.netG(self.real_B)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_B) * lambda_idt
        else:
            self.loss_idt_B = 0

        if self.opt.style_weight > 0.0:
            self.loss_style = self.opt.style_weight * self.style_loss(self.revise_B, self.real_B, hdr=False)
        else:
            self.loss_style = 0

        if self.opt.loss == 'relative':
            fake_B = self.fake_B_pool.query(self.revise_B)
            self.loss_G = self.backward_D_basic(self.netD, fake_B, self.real_B, self.opt.loss)
        else:
            # GAN loss D(G(A))
            self.loss_G = self.criterionGAN(self.netD(self.revise_B), True)


        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.revise_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE
        
        # combined loss
        self.loss_G = self.loss_G + self.loss_idt_B + self.loss_style + loss_NCE_both
        self.loss_G.backward()
    
    def optimize_parameters(self):
        # forward
        self.forward()
        # G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample' and self.opt.lambda_NCE > 0.0:
            self.optimizer_F.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample' and self.opt.lambda_NCE > 0.0:
            self.optimizer_F.step()
        # D
        for _ in range(self.d_iter):
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()
