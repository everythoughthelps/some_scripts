from PIL import Image
from matplotlib import pyplot as pt
import numpy as np
import torchvision as tv
import torch
image = '/home/panmeng/data/nyu_images/test_dir/83.png'

image_PIL = Image.open(image)
image_PIL.show()
image_np = np.array(image_PIL)
np_image = Image.fromarray(image_np)
np_image.show()
np2tensor = tv.transforms.ToTensor()(image_np)
img2tensor = tv.transforms.ToTensor()(image_PIL)
tensor2img = tv.transforms.ToPILImage()(img2tensor)
tensor2img.show()
np_tensor = torch.from_numpy(image_np)
tensor_np = np.array(np_tensor)



image_cv2_gray = cv2.imread(image,0)
image_cv2_color = cv2.imread(image,1)

cv2.imshow('gray',image_cv2_gray)
cv2.imshow('color',image_cv2_color)

cv2.waitKey(0)
cv2.destroyAllWindows()

image_plot = pt.imread(image)
pt.imshow(image_cv2_gray)
pt.show()
