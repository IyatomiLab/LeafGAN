import itertools

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from util.image_pool import ImagePool

from . import networks
from .base_model import BaseModel
from .grad_cam import GradCAM


class LeafGANModel(BaseModel):
	"""
	This class implements the LeafGAN model, for generating high-quality and diversity disease images from healthy.
	LeafGAN is basically an improved version of CycleGAN with the attention mechanism to focus on translating leaf area only.
	LeafGAN paper: https://arxiv.org/abs/2002.10100
	"""
	@staticmethod
	def modify_commandline_options(parser, is_train=True):
		"""Add new dataset-specific options, and rewrite default values for existing options.
		Parameters:
			parser          -- original option parser
			is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
		Returns:
			the modified parser.
		For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
		A (source domain), B (target domain).
		Generators: G_A: A -> B; G_B: B -> A.
		Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
		Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
		Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
		Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A)
		Dropout is not used in the LeafGAN.
		"""
		parser.set_defaults(no_dropout=True)  # default LeafGAN did not use dropout
		if is_train:
			parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
			parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
			parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

		return parser

	def __init__(self, opt):
		"""Initialize the CycleGAN class.
		Parameters:
			opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
		"""
		BaseModel.__init__(self, opt)
                self.is_using_mask = opt.dataset_mode == "unaligned_masked"
		# specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
		self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
		# specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
		visual_names_A = ['real_A', 'fake_B', 'rec_A']
		visual_names_B = ['real_B', 'fake_A', 'rec_B']
		if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
			visual_names_A.append('idt_B')
			visual_names_B.append('idt_A')

		self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
		# specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
		if self.isTrain:
			self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
		else:  # during test time, only load Gs
			self.model_names = ['G_A', 'G_B']

		# define networks (both Generators and discriminators)
		# The naming is different from those used in the paper.
		# Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
		self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
										not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
		self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
										not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

		if self.isTrain:  # define discriminators
			self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
											opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
			self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
											opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

                        if not self.is_using_mask:
                                # define the LFLSeg module
                                ######################################################
                                self.segResNet = models.resnet101()
                                num_ftrs = self.segResNet.fc.in_features
                                self.segResNet.fc = nn.Linear(
                                    num_ftrs, 3
                                )  # Replace final layer with 3 outputs (full leaf, partial leaf, non-leaf)

                                load_path = "/path/to/LFLSeg_model.pth"
                                self.segResNet.load_state_dict(torch.load(load_path), strict=True)
                                self.segResNet.to(self.device)
                                self.segResNet.eval()
                                # self.segResNet = torch.nn.DataParallel(self.segResNet, self.gpu_ids)

                                self.netLFLSeg = GradCAM(model=self.segResNet)
                                ######################################################

		if self.isTrain:
			if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
				assert(opt.input_nc == opt.output_nc)
			self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
			self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
			# define loss functions
			self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
			self.criterionCycle = torch.nn.L1Loss()
			self.criterionBackground = torch.nn.L1Loss()
			self.criterionIdt = torch.nn.L1Loss()
			# initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
			self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
			self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
			self.optimizers.append(self.optimizer_G)
			self.optimizers.append(self.optimizer_D)

	# Get the binary mask of the "full leaf" area
	def get_masking(self, tensor, threshold):
		with torch.enable_grad():
			probs, idx = self.netLFLSeg.forward(tensor)
			self.netLFLSeg.backward(idx=0) # 0 for getting heatmap for "fully_leaf" class

		heat_map = self.netLFLSeg.generate(target_layer='layer4.2') # 'layer4.2' is the best for our experiment
		heat_map = cv2.resize(heat_map, dsize=(self.opt.crop_size, self.opt.crop_size))

		background_mask = np.absolute(1.0-(heat_map>=threshold))
		background_mask = np.stack((background_mask, background_mask, background_mask), axis=2)

		foreground_mask =  heat_map>=threshold
		foreground_mask = np.stack((foreground_mask, foreground_mask, foreground_mask), axis=2)

		# from numpy image: H x W x C to torch image: C x H x W
		background_mask = background_mask.astype(np.float32).transpose(2,0,1)
		foreground_mask = foreground_mask.astype(np.float32).transpose(2,0,1)

		return torch.from_numpy(background_mask).unsqueeze(0).to(self.device), torch.from_numpy(foreground_mask).unsqueeze(0).to(self.device)

	def to_numpy(self, tensor):
		img = tensor.data
		image_numpy = img[0].cpu().float().numpy()
		image_numpy = (np.transpose(image_numpy, (1, 2, 0))  - 1.0) / 2.0 * 255.0 # Post-processing
		image_numpy = image_numpy.astype(np.uint8)

		# image_pil = Image.fromarray(image_numpy)
		# image_pil.save('whole_image.png')
		return image_numpy

	def save_image(self, tensor, filename):
		image_pil = Image.fromarray(self.to_numpy(tensor))
		image_pil.save(filename)

	def set_input(self, input):
		"""Unpack input data from the dataloader and perform necessary pre-processing steps.
		Parameters:
			input (dict): include the data itself and its metadata information.
		The option 'direction' can be used to swap domain A and domain B.
		"""
		AtoB = self.opt.direction == 'AtoB'
		self.real_A = input['A' if AtoB else 'B'].to(self.device)
		self.real_B = input['B' if AtoB else 'A'].to(self.device)
		self.image_paths = input['A_paths' if AtoB else 'B_paths']

                if self.is_using_mask:
                        self.foreground_real_A = input["mask_A" if AtoB else "mask_B"].to(self.device)
                        self.foreground_real_B = input["mask_B" if AtoB else "mask_A"].to(self.device)
                        with torch.no_grad():
                                self.background_real_A = torch.absolute(1.0 - self.foreground_real_A)
                                self.background_real_B = torch.absolute(1.0 - self.foreground_real_B)


	def forward(self):
		"""Run forward pass; called by both functions <optimize_parameters> and <test>."""
		# For training
		if(self.isTrain):
                        if not self.is_using_mask:
                                self.background_real_A, self.foreground_real_A = self.get_masking(
                                    self.real_A, self.opt.threshold
                                )
                                self.background_real_B, self.foreground_real_B = self.get_masking(
                                    self.real_B, self.opt.threshold
                                )
                                # To save the segmented results, use save_image
                                # self.save_image(self.background_real_A, 'saved_img/masked_background_real_A.png')
                                # self.save_image(self.foreground_real_A, 'saved_img/masked_foreground_real_A.png')

			# Fore real_A input
			self.fake_B = self.netG_A(self.real_A)  # G_A(A)

			# multyple the fore/baclground masking of real_A to fake_B to get fore/background of fake_B
			self.fore_fake_B = self.foreground_real_A * self.fake_B
			self.back_fake_B = self.background_real_A * self.fake_B

			self.fore_real_B = self.foreground_real_B * self.real_B
			self.back_real_B = self.background_real_B * self.real_B

			self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))

			# For real_B input
			self.fake_A = self.netG_B(self.real_B)  # G_B(B)

			# multyple the fore/baclground masking of real_B to fake_A to get fore/background of fake_A
			self.fore_fake_A = self.foreground_real_B * self.fake_A
			self.back_fake_A = self.background_real_B * self.fake_A

			self.fore_real_A = self.foreground_real_A * self.real_A
			self.back_real_A = self.background_real_A * self.real_A

			self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))
		# For testing, no need to load LFLSeg module
		else:
			# Fore real_A input
			self.fake_B = self.netG_A(self.real_A)  # G_A(A)
			self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))

			# For real_B input
			self.fake_A = self.netG_B(self.real_B)  # G_B(B)
			self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

	def backward_D_basic(self, netD, real, fake):
		"""Calculate GAN loss for the discriminator
		Parameters:
			netD (network)      -- the discriminator D
			real (tensor array) -- real images
			fake (tensor array) -- images generated by a generator
		Return the discriminator loss.
		We also call loss_D.backward() to calculate the gradients.
		"""
		# Real
		pred_real = netD(real)
		loss_D_real = self.criterionGAN(pred_real, True)
		# Fake
		pred_fake = netD(fake.detach())
		loss_D_fake = self.criterionGAN(pred_fake, False)
		# Combined loss and calculate gradients
		loss_D = (loss_D_real + loss_D_fake) * 0.5
		loss_D.backward()
		return loss_D

	def backward_D_A(self):
		"""Calculate GAN loss for discriminator D_A"""
		fore_fake_B = self.fake_B_pool.query(self.fore_fake_B)
		# Input of the discriminator is only leaf area
		self.loss_D_A = self.backward_D_basic(self.netD_A, self.fore_real_B, fore_fake_B)

	def backward_D_B(self):
		"""Calculate GAN loss for discriminator D_B"""
		fore_fake_A = self.fake_A_pool.query(self.fore_fake_A)
		# Input of the discriminator is only leaf area
		self.loss_D_B = self.backward_D_basic(self.netD_B, self.fore_real_A, fore_fake_A)

	def backward_G(self):
		"""Calculate the loss for generators G_A and G_B"""
		lambda_idt = self.opt.lambda_identity
		lambda_A = self.opt.lambda_A
		lambda_B = self.opt.lambda_B
		# Identity loss
		if lambda_idt > 0:
			# G_A should be identity if real_B is fed: ||G_A(B) - B||
			self.idt_A = self.netG_A(self.real_B)
			self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
			# G_B should be identity if real_A is fed: ||G_B(A) - A||
			self.idt_B = self.netG_B(self.real_A)
			self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
		else:
			self.loss_idt_A = 0
			self.loss_idt_B = 0

		# GAN loss D_A(G_A(A))
		self.loss_G_A = self.criterionGAN(self.netD_A(self.fore_fake_B), True)
		# GAN loss D_B(G_B(B))
		self.loss_G_B = self.criterionGAN(self.netD_B(self.fore_fake_A), True)
		# Forward cycle loss || G_B(G_A(A)) - A||
		self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
		# Backward cycle loss || G_A(G_B(B)) - B||
		self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

		# Forward background loss
		self.loss_background_A = self.criterionBackground(self.back_fake_B, self.back_real_A) * lambda_A * lambda_idt
		# Backward background loss
		self.loss_background_B = self.criterionBackground(self.back_fake_A, self.back_real_B) * lambda_B * lambda_idt

		# combined loss and calculate gradients
		self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_background_A + self.loss_background_B
		self.loss_G.backward()

	def optimize_parameters(self):
		"""Calculate losses, gradients, and update network weights; called in every training iteration"""
		# forward
		self.forward()      # compute fake images and reconstruction images.
		# G_A and G_B
		self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
		self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
		self.backward_G()             # calculate gradients for G_A and G_B
		self.optimizer_G.step()       # update G_A and G_B's weights
		# D_A and D_B
		self.set_requires_grad([self.netD_A, self.netD_B], True)
		self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
		self.backward_D_A()      # calculate gradients for D_A
		self.backward_D_B()      # calculate graidents for D_B
		self.optimizer_D.step()  # update D_A and D_B's weights
