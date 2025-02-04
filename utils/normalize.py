import cv2
import numpy as np
import torch
from torch.autograd import Variable

def preprocess_image(cv2im, resize_im=True):
    if resize_im:
        cv2im = cv2.resize(cv2im, (224, 224))
    im_as_arr = np.float32(cv2im)
    im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
    im_as_arr = im_as_arr.transpose(2, 0, 1)
    im_as_ten = torch.from_numpy(im_as_arr).float()
    im_as_ten = im_as_ten.unsqueeze(0)
    im_as_var = Variable(im_as_ten, requires_grad=True)
    visualize_image = im_as_arr.transpose(1, 2, 0)
    return im_as_ten, im_as_var, visualize_image