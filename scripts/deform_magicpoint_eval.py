import os, sys
import numpy as np
import torch
from tqdm import tqdm
import yaml

from Val_model_subpixel import Val_model_subpixel
from models.DSuperPointNet_gauss2 import main as test_module

filename = 'configs/magicpoint_eval.yaml'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_default_tensor_type(torch.FloatTensor)
with open(filename, 'r') as f:
    config = yaml.load(f,Loader=yaml.FullLoader)

# data loading
from utils.loader import dataLoader_test as dataLoader
task = config['data']['dataset']
data = dataLoader(config, dataset=task)
test_set, test_loader = data['test_set'], data['test_loader']

from utils.print_tool import datasize
datasize(test_loader, config, tag='test')

# model loading
from utils.loader import get_module
Val_model_heatmap = get_module('', config['front_end_model'])

## load pretrained
val_agent = Val_model_heatmap(config['model'], device=device)
val_agent.loadModel()

net = val_agent.net
# load data
for i, sample in tqdm(enumerate(test_loader)):
    # print(sample.keys())
    # img_0, img_1 = sample['image'], sample['warped_image']
    img = sample['image']
    # print(img.shape)
    
    # if i>1: break

    img = img.permute((0,3,1,2))
    outs = net(img.to(device))
    print("outs: ", list(outs))


    # process outputs
    from utils.print_tool import print_dict_attr
    print_dict_attr(outs, 'shape')

    from models.model_utils import SuperPointNet_process 
    params = {
        'out_num_points': 500,
        'patch_size': 5,
        'device': device,
        'nms_dist': 4,
        'conf_thresh': 0.015
    }

    sp_processer = SuperPointNet_process(**params)
    outs_post = net.process_output(sp_processer)

    print("outs: ", list(outs_post))
    from utils.print_tool import print_dict_attr
    print_dict_attr(outs_post, 'shape')


    pts_int = outs_post['pts_int']
    pts_offset = outs_post['pts_offset']
    pts_desc = outs_post['pts_desc']
    print("pts_offset: ", pts_offset[0, :5, :])
    def print_attr(dictionary, attr):
        for en in list(dictionary):
            print(en, attr, ": ", getattr(dictionary[en], attr))
    print_attr(outs_post, 'requires_grad')

    from utils.draw import draw_keypoints
    from utils.utils import toNumpy
    import matplotlib.pyplot as plt

    # for i in range(2):
    img = draw_keypoints(toNumpy(img.squeeze()), toNumpy((pts_int+pts_offset).squeeze()).transpose())
    # print("img: ", img_0)
    plt.imshow(img)
    plt.show()