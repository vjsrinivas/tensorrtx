import struct
import sys
from models.yolo import Model
import yaml
from utils.datasets import *
#from utils.utils import *

# change to argparse
weight_file = sys.argv[1]
output = sys.argv[2]

model = torch.load(weight_file, map_location="cuda:0")  # load to FP32
model = model['ema' if model.get('ema') else 'model'].float()

# update anchor_grid info
#anchor_grid = model.model[-1].anchors * model.model[-1].stride[...,None,None]
#model.model[-1].anchor_grid = anchor_grid
#delattr(model.model[-1], 'anchor_grid')  # model.model[-1] is detect layer
#model.model[-1].register_buffer("anchor_grid",anchor_grid) #The parameters are saved in the OrderDict through the "register_buffer" method, and then saved to the weight.

model.to("cuda:0").eval()
#print(model)

true_idx = 0
_p = 0
_p2 = 0
padding = 0
anchor_pt = 347
anchor_loc = [347, 391, 435]
anchor_loc_2 = [435, 391, 347]
dict_len = len(model.state_dict().keys()) - 2 # -2 for anchor 
#padding = 0

idx_key = {} # weight key and wanted index:

yolo_conv_count = 0
og_yolo_conv = 21

'''
for i, (k, v) in enumerate(model.state_dict().items()):
    # check for yolo-conv layers:
    _k = k.split('.')
    if 'm' in _k:
        # it is!
        _k[1] = int(_k[1])
        yolo_conv_count += 1
        k = "module_list.%s.conv.%s"%(og_yolo_conv, _k[4])
        if yolo_conv_count % 2 == 0:
            og_yolo_conv += 1
    else:
        if len(_k) > 3:
            k = "module_list.%s.%s.%s"%(_k[1], _k[2], _k[3])
        else:
            k = "module_list.%s.%s"%(_k[1], _k[2],)
'''

with open(output, 'w') as f:
    f.write('{}\n'.format(len(model.state_dict().keys())))
    for i, (k, v) in enumerate(model.state_dict().items()):
        vr = v.reshape(-1).cpu().numpy()
    
        # check for yolo-conv layers:
        _k = k.split('.')
        if 'm' in _k:
            # it is!
            _k[1] = int(_k[1])
            yolo_conv_count += 1
            k = "module_list.%s.conv.%s"%(og_yolo_conv, _k[4])
            if yolo_conv_count % 2 == 0:
                og_yolo_conv += 1
        else:
            if len(_k) > 3:
                k = "module_list.%s.%s.%s"%(_k[1], _k[2], _k[3])
            else:
                k = "module_list.%s.%s"%(_k[1], _k[2],)
        print(i, k, vr.shape, v.shape)

        f.write('{} {}'.format(
            k,
            len(vr))    
        )

        for vv in vr:
            f.write(' ')
            f.write(struct.pack('>f' ,float(vv)).hex())
        f.write('\n')
