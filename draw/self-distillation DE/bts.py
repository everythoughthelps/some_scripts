# Copyright (C) 2019 Jin Han Lee
#
# This file is a part of BTS.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

import torch
import torch.nn as nn
import torch.nn.functional as torch_nn_func
import math

from collections import namedtuple


# This sets the batch norm layers in pytorch as if {'is_training': False, 'scale': True} in tensorflow
def bn_init_as_tf(m):
	if isinstance(m, nn.BatchNorm2d):
		m.track_running_stats = True  # These two lines enable using stats (moving mean and var) loaded from pretrained model
		m.eval()                      # or zero mean and variance of one if the batch norm layer has no pretrained values
		m.affine = True
		m.requires_grad = True


def weights_init_xavier(m):
	if isinstance(m, nn.Conv2d):
		torch.nn.init.xavier_uniform_(m.weight)
		if m.bias is not None:
			torch.nn.init.zeros_(m.bias)


class silog_loss(nn.Module):
	def __init__(self, variance_focus):
		super(silog_loss, self).__init__()
		self.variance_focus = variance_focus

	def forward(self, depth_est, depth_gt, mask):
		d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
		return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0


class atrous_conv(nn.Sequential):
	def __init__(self, in_channels, out_channels, dilation, apply_bn_first=True):
		super(atrous_conv, self).__init__()
		self.atrous_conv = torch.nn.Sequential()
		if apply_bn_first:
			self.atrous_conv.add_module('first_bn', nn.BatchNorm2d(in_channels, momentum=0.01, affine=True, track_running_stats=True, eps=1.1e-5))

		self.atrous_conv.add_module('aconv_sequence', nn.Sequential(nn.ReLU(),
																	nn.Conv2d(in_channels=in_channels, out_channels=out_channels*2, bias=False, kernel_size=1, stride=1, padding=0),
																	nn.BatchNorm2d(out_channels*2, momentum=0.01, affine=True, track_running_stats=True),
																	nn.ReLU(),
																	nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, bias=False, kernel_size=3, stride=1,
																			  padding=(dilation, dilation), dilation=dilation)))

	def forward(self, x):
		return self.atrous_conv.forward(x)


class upconv(nn.Module):
	def __init__(self, in_channels, out_channels, ratio=2):
		super(upconv, self).__init__()
		self.elu = nn.ELU()
		self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, bias=False, kernel_size=3, stride=1, padding=1)
		self.ratio = ratio

	def forward(self, x):
		up_x = torch_nn_func.interpolate(x, scale_factor=self.ratio, mode='nearest')
		out = self.conv(up_x)
		out = self.elu(out)
		return out


class reduction_1x1(nn.Sequential):
	def __init__(self, num_in_filters, num_out_filters, is_final=False):
		super(reduction_1x1, self).__init__()
		self.is_final = is_final
		self.sigmoid = nn.Sigmoid()
		self.reduc = torch.nn.Sequential()

		while num_out_filters >= 4:
			if num_out_filters < 8:
				if self.is_final:
					self.reduc.add_module('final', torch.nn.Sequential(nn.Conv2d(num_in_filters, out_channels=1, bias=False,
																				 kernel_size=1, stride=1, padding=0),
																	   nn.Sigmoid()))
				else:
					self.reduc.add_module('plane_params', torch.nn.Conv2d(num_in_filters, out_channels=3, bias=False,
																		  kernel_size=1, stride=1, padding=0))
				break
			else:
				self.reduc.add_module('inter_{}_{}'.format(num_in_filters, num_out_filters),
									  torch.nn.Sequential(nn.Conv2d(in_channels=num_in_filters, out_channels=num_out_filters,
																	bias=False, kernel_size=1, stride=1, padding=0),
														  nn.ELU()))

			num_in_filters = num_out_filters
			num_out_filters = num_out_filters // 2

	def forward(self, net,max_depth=None):
		net = self.reduc.forward(net)
		if not self.is_final:
			theta = self.sigmoid(net[:, 0, :, :]) * math.pi / 3
			phi = self.sigmoid(net[:, 1, :, :]) * math.pi * 2
			dist = self.sigmoid(net[:, 2, :, :]) * max_depth[0]
			n1 = torch.mul(torch.sin(theta), torch.cos(phi)).unsqueeze(1)
			n2 = torch.mul(torch.sin(theta), torch.sin(phi)).unsqueeze(1)
			n3 = torch.cos(theta).unsqueeze(1)
			n4 = dist.unsqueeze(1)
			net = torch.cat([n1, n2, n3, n4], dim=1)

		return net

class local_planar_guidance(nn.Module):
	def __init__(self, upratio):
		super(local_planar_guidance, self).__init__()
		self.upratio = upratio
		self.u = torch.arange(self.upratio).reshape([1, 1, self.upratio]).float()
		self.v = torch.arange(int(self.upratio)).reshape([1, self.upratio, 1]).float()
		self.upratio = float(upratio)

	def forward(self, plane_eq, focal):
		plane_eq_expanded = torch.repeat_interleave(plane_eq, int(self.upratio), 2)
		plane_eq_expanded = torch.repeat_interleave(plane_eq_expanded, int(self.upratio), 3)
		n1 = plane_eq_expanded[:, 0, :, :]
		n2 = plane_eq_expanded[:, 1, :, :]
		n3 = plane_eq_expanded[:, 2, :, :]
		n4 = plane_eq_expanded[:, 3, :, :]

		u = self.u.repeat(plane_eq.size(0), plane_eq.size(2) * int(self.upratio), plane_eq.size(3)).cuda()
		u = (u - (self.upratio - 1) * 0.5) / self.upratio

		v = self.v.repeat(plane_eq.size(0), plane_eq.size(2), plane_eq.size(3) * int(self.upratio)).cuda()
		v = (v - (self.upratio - 1) * 0.5) / self.upratio

		return n4 / (n1 * u + n2 * v + n3)

class bts(nn.Module):
	def __init__(self, params, feat_out_channels, num_features=512):
		super(bts, self).__init__()
		self.params = params

		self.upconv5    = upconv(feat_out_channels[4], num_features)
		self.bn5        = nn.BatchNorm2d(num_features, momentum=0.01, affine=True, eps=1.1e-5)

		self.conv5      = torch.nn.Sequential(nn.Conv2d(num_features + feat_out_channels[3], num_features, 3, 1, 1, bias=False),
											  nn.ELU())
		self.upconv4    = upconv(num_features, num_features // 2)
		self.bn4        = nn.BatchNorm2d(num_features // 2, momentum=0.01, affine=True, eps=1.1e-5)
		self.conv4      = torch.nn.Sequential(nn.Conv2d(num_features // 2 + feat_out_channels[2], num_features // 2, 3, 1, 1, bias=False),
											  nn.ELU())
		self.bn4_2      = nn.BatchNorm2d(num_features // 2, momentum=0.01, affine=True, eps=1.1e-5)

		self.daspp_3    = atrous_conv(num_features // 2, num_features // 4, 3, apply_bn_first=False)
		self.daspp_6    = atrous_conv(num_features // 2 + num_features // 4 + feat_out_channels[2], num_features // 4, 6)
		self.daspp_12   = atrous_conv(num_features + feat_out_channels[2], num_features // 4, 12)
		self.daspp_18   = atrous_conv(num_features + num_features // 4 + feat_out_channels[2], num_features // 4, 18)
		self.daspp_24   = atrous_conv(num_features + num_features // 2 + feat_out_channels[2], num_features // 4, 24)
		self.daspp_conv = torch.nn.Sequential(nn.Conv2d(num_features + num_features // 2 + num_features // 4, num_features // 4, 3, 1, 1, bias=False),
											  nn.ELU())
		self.reduc8x8   = reduction_1x1(num_features // 4, num_features // 4)
		self.lpg8x8     = local_planar_guidance(8)

		self.upconv3    = upconv(num_features // 4, num_features // 4)
		self.bn3        = nn.BatchNorm2d(num_features // 4, momentum=0.01, affine=True, eps=1.1e-5)
		self.conv3      = torch.nn.Sequential(nn.Conv2d(num_features // 4 + feat_out_channels[1] + 1, num_features // 4, 3, 1, 1, bias=False),
											  nn.ELU())
		self.reduc4x4   = reduction_1x1(num_features // 4, num_features // 8)
		self.lpg4x4     = local_planar_guidance(4)

		self.upconv2    = upconv(num_features // 4, num_features // 8)
		self.bn2        = nn.BatchNorm2d(num_features // 8, momentum=0.01, affine=True, eps=1.1e-5)
		self.conv2      = torch.nn.Sequential(nn.Conv2d(num_features // 8 + feat_out_channels[0] + 1, num_features // 8, 3, 1, 1, bias=False),
											  nn.ELU())

		self.reduc2x2   = reduction_1x1(num_features // 8, num_features // 16)
		self.lpg2x2     = local_planar_guidance(2)

		self.upconv1    = upconv(num_features // 8, num_features // 16)
		self.reduc1x1   = reduction_1x1(num_features // 16, num_features // 32, is_final=True)
		self.conv1      = torch.nn.Sequential(nn.Conv2d(num_features // 16 + 4, num_features // 16, 3, 1, 1, bias=False),
											  nn.ELU())
		self.get_depth  = torch.nn.Sequential(nn.Conv2d(num_features // 16, 1, 3, 1, 1, bias=False),
											  nn.Sigmoid())

	def forward(self, features, focal,maxdepth):
		skip0, skip1, skip2, skip3 = features[1], features[2], features[3], features[4]
		dense_features = torch.nn.ReLU()(features[5])
		upconv5 = self.upconv5(dense_features) # H/16
		upconv5 = self.bn5(upconv5)
		concat5 = torch.cat([upconv5, skip3], dim=1)
		iconv5 = self.conv5(concat5)

		upconv4 = self.upconv4(iconv5) # H/8
		upconv4 = self.bn4(upconv4)
		concat4 = torch.cat([upconv4, skip2], dim=1)
		iconv4 = self.conv4(concat4)
		iconv4 = self.bn4_2(iconv4)

		daspp_3 = self.daspp_3(iconv4)
		concat4_2 = torch.cat([concat4, daspp_3], dim=1)
		daspp_6 = self.daspp_6(concat4_2)
		concat4_3 = torch.cat([concat4_2, daspp_6], dim=1)
		daspp_12 = self.daspp_12(concat4_3)
		concat4_4 = torch.cat([concat4_3, daspp_12], dim=1)
		daspp_18 = self.daspp_18(concat4_4)
		concat4_5 = torch.cat([concat4_4, daspp_18], dim=1)
		daspp_24 = self.daspp_24(concat4_5)
		concat4_daspp = torch.cat([iconv4, daspp_3, daspp_6, daspp_12, daspp_18, daspp_24], dim=1)
		daspp_feat = self.daspp_conv(concat4_daspp)
		unify_out = torch.nn.functional.adaptive_avg_pool2d(daspp_feat,(44,68))

		return unify_out

class encoder(nn.Module):
	def __init__(self, params):
		super(encoder, self).__init__()
		self.params = params
		import torchvision.models as models
		if params.encoder == 'densenet121_bts':
			self.base_model = models.densenet121(pretrained=True).features
			self.feat_names = ['relu0', 'pool0', 'transition1', 'transition2', 'norm5']
			self.feat_out_channels = [64, 64, 128, 256, 1024]
		elif params.encoder == 'densenet161_bts':
			self.base_model = models.densenet161(pretrained=True).features
			self.feat_names = ['relu0', 'pool0', 'transition1', 'transition2', 'norm5']
			self.feat_out_channels = [96, 96, 192, 384, 2208]
		elif params.encoder == 'resnet50_bts':
			self.base_model = models.resnet50(pretrained=True)
			self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
			self.feat_out_channels = [64, 256, 512, 1024, 2048]
		elif params.encoder == 'resnet101_bts':
			self.base_model = models.resnet101(pretrained=True)
			self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
			self.feat_out_channels = [64, 256, 512, 1024, 2048]
		elif params.encoder == 'resnext50_bts':
			self.base_model = models.resnext50_32x4d(pretrained=True)
			self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
			self.feat_out_channels = [64, 256, 512, 1024, 2048]
		elif params.encoder == 'resnext101_bts':
			self.base_model = models.resnext101_32x8d(pretrained=True)
			self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
			self.feat_out_channels = [64, 256, 512, 1024, 2048]
		else:
			print('Not supported encoder: {}'.format(params.encoder))

	def forward(self, x):
		features = [x]
		skip_feat = [x]
		for k, v in self.base_model._modules.items():
			if 'fc' in k or 'avgpool' in k:
				continue
			feature = v(features[-1])
			features.append(feature)
			if any(x in k for x in self.feat_names):
				skip_feat.append(feature)

		return skip_feat


class BtsModel(nn.Module):
	def __init__(self, params):
		super(BtsModel, self).__init__()
		self.encoder = encoder(params)
		self.decoder = bts(params, self.encoder.feat_out_channels, params.bts_size)

	def forward(self, x, focal,maxdepth):
		skip_feat = self.encoder(x)
		return self.decoder(skip_feat, focal,maxdepth)
