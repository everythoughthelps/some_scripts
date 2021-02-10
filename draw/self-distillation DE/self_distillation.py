import argparse
import random

import seaborn as sns
from bts import *
from PIL import Image
from matplotlib import pyplot as plt
import torch
import numpy as np

parser = argparse.ArgumentParser(description='BTS PyTorch implementation.', fromfile_prefix_chars='@')
parser.add_argument('--encoder', type=str, help='type of encoder, vgg or desenet121_bts or densenet161_bts',
					default='densenet161_bts')
parser.add_argument('--bts_size', type=int,   help='initial num_filters in bts', default=512)

args = parser.parse_args()

model_path = '/home/panmeng/PycharmProjects/bts/bts/' \
			 'pytorch/models_base_unify/kitti/bts_v2_pytorch_test/models-79500-best_rms_3.25714'
our_model_path = '/home/panmeng/PycharmProjects/bts/bts/' \
				 'pytorch/models_1/kitti/bts_v2_pytorch_test/models-51500-best_rms_3.16750'

checkpoint = torch.load(model_path)
model = checkpoint['models']
weight = model['module.decoder.daspp_conv.0.weight']

our_checkpoint = torch.load(our_model_path)
our_model = our_checkpoint['models']
our_weight = our_model['module.decoder.daspp_conv.0.weight']

def seabron_func(weight,our_weight):
	flatten_weight = torch.flatten(weight, start_dim=1)
	our_flatten_weight = torch.flatten(our_weight, start_dim=1)
	matrix_co = np.corrcoef(flatten_weight.cpu())
	our_matrix_co = np.corrcoef(our_flatten_weight.cpu())
	cmap = 'jet'
	mask = np.zeros_like(matrix_co)
	index = np.triu_indices_from(mask)
	mask[index] = True


	f,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
	sns.heatmap(matrix_co,mask=mask,ax=ax1,cmap=cmap,square=True,xticklabels=20,yticklabels=20)
	ax1.set_xlim(0,127)
	ax1.set_ylim(127,0)
	sns.heatmap(our_matrix_co,mask=mask,ax=ax2,cmap=cmap,square=True,xticklabels=20,yticklabels=20)
	ax2.set_xlim(0,127)
	ax2.set_ylim(127,0)
	#plt.axis('off')
	plt.tight_layout(pad=0)
	plt.savefig('matrix_origin.pdf',dpi=300,bbox_inches='tight',pad_inches=0)
	plt.show()

def imshow_fun(weight,our_weight):
	flatten_weight = torch.flatten(weight, start_dim=1)
	our_flatten_weight = torch.flatten(our_weight, start_dim=1)

	matrix_co = np.corrcoef(flatten_weight.cpu())
	our_matrix_co = np.corrcoef(our_flatten_weight.cpu())
	mask = np.zeros_like(matrix_co)
	mask[np.triu_indices_from(mask)] = True
	matrix_co_tridown = np.ma.masked_where(mask,matrix_co)
	our_matrix_co_tridown = np.ma.masked_where(mask,our_matrix_co)
	vmax = max(our_matrix_co_tridown.max(),matrix_co_tridown.max())
	vmin = min(our_matrix_co_tridown.min(),matrix_co_tridown.min())

	fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
	axs[0].spines['right'].set_visible(False)
	axs[0].spines['top'].set_visible(False)
	axs[0].spines['bottom'].set_visible(False)
	axs[0].spines['left'].set_visible(False)
	axs[1].spines['right'].set_visible(False)
	axs[1].spines['top'].set_visible(False)
	axs[1].spines['bottom'].set_visible(False)
	axs[1].spines['left'].set_visible(False)
	sc0=axs[0].imshow(matrix_co_tridown, cmap='jet',vmax=vmax,vmin=vmin)
	sc1=axs[1].imshow(our_matrix_co_tridown, cmap='jet',vmax=vmax,vmin=vmin)

	fig.subplots_adjust(right=0.9)
	# colorbar 左 下 宽 高
	l = 0.92
	b = 0.23
	w = 0.015
	h = 0.53
	# 对应 l,b,w,h；设置colorbar位置；
	rect = [l, b, w, h]
	cbar_ax = fig.add_axes(rect)
	plt.colorbar(sc0,cax=cbar_ax)
	plt.savefig('conv_c.pdf', dpi=300, bbox_inches='tight')
	plt.show()

class plt_hist(object):
	def __init__(self,rows,clos):
		self.fig, self.axs = plt.subplots(nrows=rows, ncols=clos, figsize=(10, 4))

	def histgram(self, mat_x, axe, mat_y=None):
		if mat_y is not None:
			flatten_mat_x = torch.flatten(mat_x, start_dim=2).squeeze()
			flatten_mat_y = torch.flatten(mat_y, start_dim=2).squeeze()
			matrix_co = np.corrcoef(flatten_mat_x.cpu(),flatten_mat_y.cpu())
		else:
			flatten_mat_x = torch.flatten(mat_x, start_dim=2).squeeze()
			matrix_co = np.corrcoef(flatten_mat_x.cpu())
		mask = np.zeros_like(matrix_co)
		mask[np.triu_indices_from(mask)] = True
		matrix_co_tridown = np.ma.masked_where(mask, matrix_co)
		axe.set_xlabel('Pearson correlation coefficient')
		axe.set_ylabel('number of elements')
		axe.set_xticklabels(labels = ['0','-0.5','0.0','0.5','1.0'], fontdict={'size':8})
		axe.set_yticks(range(0,13000,2000))
		axe.set_yticklabels(labels = ['0','2000','4000','6000','8000','10000','12000'], fontdict={'size':8})
		axe.spines['right'].set_visible(False)
		axe.spines['top'].set_visible(False)
		#axe.spines['bottom'].set_visible(False)
		axe.spines['left'].set_visible(False)
		axe.grid(True,linestyle='--',alpha = 0.4,axis = 'y')
		arr = axe.hist(matrix_co_tridown.ravel(),bins=20)
		#for i in range(20):
		#	axe.text(arr[1][i], arr[0][i], str(arr[0][i]))

	def imshow_fun(self,weight,axe,our_weight=None):
		if our_weight is not None:
			flatten_weight = torch.flatten(weight, start_dim=1)
			our_flatten_weight = torch.flatten(our_weight, start_dim=1)

			matrix_co = np.corrcoef(flatten_weight.cpu(),our_flatten_weight.cpu())
		else:
			flatten_weight = torch.flatten(weight, start_dim=1)
			matrix_co = np.corrcoef(flatten_weight.cpu())

		mask = np.zeros_like(matrix_co)
		mask[np.triu_indices_from(mask)] = True
		matrix_co_tridown = np.ma.masked_where(mask,matrix_co)

		axe.spines['right'].set_visible(False)
		axe.spines['top'].set_visible(False)
		axe.spines['bottom'].set_visible(False)
		axe.spines['left'].set_visible(False)
		axe.imshow(matrix_co_tridown, cmap='jet')

	def hot_mul2one(self,weight,our_weight,axe):
		flatten_weight = torch.flatten(weight, start_dim=1)
		our_flatten_weight = torch.flatten(our_weight, start_dim=1)

		matrix_co = np.corrcoef(flatten_weight.cpu())
		our_matrix_co = np.corrcoef(our_flatten_weight.cpu())
		mask = np.zeros_like(matrix_co)
		mask[np.triu_indices_from(mask)] = True
		matrix_co_tridown = np.ma.masked_where(mask,matrix_co)
		our_matrix_co_tridown = np.ma.masked_where(mask,our_matrix_co)
		vmax = max(our_matrix_co_tridown.max(),matrix_co_tridown.max())
		vmin = min(our_matrix_co_tridown.min(),matrix_co_tridown.min())

		axe[0].spines['right'].set_visible(False)
		axe[0].spines['top'].set_visible(False)
		axe[0].spines['bottom'].set_visible(False)
		axe[0].spines['left'].set_visible(False)
		axe[1].spines['right'].set_visible(False)
		axe[1].spines['top'].set_visible(False)
		axe[1].spines['bottom'].set_visible(False)
		axe[1].spines['left'].set_visible(False)
		axe[0].set_xlabel('filter index')
		axe[0].set_ylabel('filter index')
		axe[1].set_xlabel('filter index')
		axe[1].set_ylabel('filter index')
		axe[0].set_xticks(range(0,128,20))
		axe[0].set_xticklabels(['0','20','40','60','80','100','120'],fontdict={'size':8})
		axe[1].set_xticks(range(0,128,20))
		axe[1].set_xticklabels(['0','20','40','60','80','100','120'],fontdict={'size':8})
		axe[0].set_yticks(range(0,128,20))
		axe[0].set_yticklabels(['0','20','40','60','80','100','120'],fontdict={'size':8})
		axe[1].set_yticks(range(0,128,20))
		axe[1].set_yticklabels(['0','20','40','60','80','100','120'],fontdict={'size':8})
		axe[0].set_title('(a)',y=-0.2,fontsize=10)
		axe[1].set_title('(b)',y=-0.2,fontsize=10)
		sc0=axe[0].imshow(matrix_co_tridown, cmap='jet',vmax=vmax,vmin=vmin)
		sc1=axe[1].imshow(our_matrix_co_tridown, cmap='jet',vmax=vmax,vmin=vmin)

		self.fig.subplots_adjust(right=0.9)
		# colorbar 左 下 宽 高
		l = 0.92
		b = 0.15
		w = 0.01
		h = 0.7
		# 对应 l,b,w,h；设置colorbar位置；
		rect = [l, b, w, h]
		cbar_ax = self.fig.add_axes(rect)
		cb = plt.colorbar(sc0,cax=cbar_ax)
		cb.ax.tick_params(labelsize=8)
		font = {#'family': 'serif',
				#'color': 'darkred',
				#'weight': 'normal',
				'size': 10,
				}
		cb.set_label('Cosine', fontdict=font)  # 设置colorbar的标签字体及其大小

def to_tensor(pic):
	if isinstance(pic, np.ndarray):
		img = torch.from_numpy(pic.transpose((2, 0, 1)))
		return img

def random_crop(img, height, width):
	assert img.shape[0] >= height
	assert img.shape[1] >= width
	x = random.randint(0, img.shape[1] - width)
	y = random.randint(0, img.shape[0] - height)
	img = img[y:y + height, x:x + width, :]
	return img

def test_feature(model_weight, img, dataset):
	image = Image.open(img)
	if dataset == 'nyu':
		image = image.crop((43, 45, 608, 472))
		image = np.asarray(image, dtype=np.float32) / 255.0
		image = random_crop(image,416,544)
	elif dataset == 'kitti':
		height = image.height
		width = image.width
		top_margin = int(height - 352)
		left_margin = int((width - 1216) / 2)
		image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
		image = np.asarray(image, dtype=np.float32) / 255.0
		image = random_crop(image, 352, 704)

	image = to_tensor(image)

	image = image.unsqueeze(0)
	model = BtsModel(args)
	ckp = torch.load(model_weight)
	model.load_state_dict({k.replace('module.',''):v for k,v in ckp['models'].items()})
	model.eval()
	model.cuda()

	with torch.no_grad():
		image = image.cuda()
		focal = torch.tensor(0.5).cuda()
		maxdepth = torch.tensor(0.5).float().cuda()
		# Predict
		unified_feature = model(image, focal, maxdepth)
	return unified_feature

nyu_img_25 = '/data/nyu/data/demo/train_image/25.png'
nyu_img_26 = '/data/nyu/data/demo/train_image/26.png'
kitti_0 = '/data/kitti/kitti_raw_data/2011_10_03/2011_10_03_drive_0047_sync/image_02/data/0000000000.png'
kitti_99 = '/data/kitti/kitti_raw_data/2011_10_03/2011_10_03_drive_0047_sync/image_02/data/0000000099.png'

nyu_feature_25 = test_feature(model_path,nyu_img_25,'nyu')
our_nyu_feature_25 = test_feature(our_model_path,nyu_img_26,'nyu')
kitti_feature_0 = test_feature(model_path,kitti_0,'kitti')
our_kitti_feature_0 = test_feature(our_model_path,kitti_0,'kitti')

a = plt_hist(1,2)
a.hot_mul2one(weight,our_weight,a.axs)
plt.savefig('conv_co.pdf', dpi=300, bbox_inches='tight')
plt.show()
'''
a = plt_hist(1,2)
a.axs[0].get_shared_y_axes().join(a.axs[0],a.axs[1])
a.axs[0].get_shared_x_axes().join(a.axs[0],a.axs[1])
a.histgram(nyu_feature_25,a.axs[0],kitti_feature_0)
a.histgram(our_nyu_feature_25,a.axs[1],our_kitti_feature_0)
a.axs[0].set_title('(a)', y=-0.2, fontsize=10)
a.axs[1].set_title('(b)', y=-0.2, fontsize=10)

plt.savefig('fea_co.pdf', dpi=300, bbox_inches='tight')
plt.show()
'''
