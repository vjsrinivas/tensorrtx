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
anchor_grid = model.model[-1].anchors * model.model[-1].stride[...,None,None]
# model.model[-1].anchor_grid = anchor_grid
delattr(model.model[-1], 'anchor_grid')  # model.model[-1] is detect layer
model.model[-1].register_buffer("anchor_grid",anchor_grid) #The parameters are saved in the OrderDict through the "register_buffer" method, and then saved to the weight.

model.to("cuda:0").eval()
print(model)

with open(output, 'w') as f:
    f.write('{}\n'.format(len(model.state_dict().keys())))
    for k, v in model.state_dict().items():
        vr = v.reshape(-1).cpu().numpy()
        f.write('{} {} '.format(k, len(vr)))
        for vv in vr:
            f.write(' ')
            f.write(struct.pack('>f' ,float(vv)).hex())
        f.write('\n')
