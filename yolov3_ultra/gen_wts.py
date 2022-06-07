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


for i, (k, v) in enumerate(model.state_dict().items()):
    vr = v.reshape(-1).cpu().numpy()
   
    if "anchor" in k:
        _p += 1
        continue

    if i > dict_len-6:
        left_over_idx = i-(dict_len-5)
        #print(padding, left_over_idx)
        idx_key[k] = true_idx+padding
        if left_over_idx % 2 == 0:
            padding += 1
        continue

    if len(anchor_loc) > 0:
        if anchor_loc[0] == i:
            _p2 += 2
            anchor_loc.pop(0)
    
    i -= _p

    #print(i, k, vr.shape)
    if not k in idx_key:
        idx_key[k] = true_idx 

    if (i+1)%6 == 0: 
        true_idx += 1
        #print("====================================")
    #print(i, true_idx, k, vr.shape)


#for i,(j,k) in enumerate(idx_key.items()):
#    print(i, k, j)

#exit()

with open(output, 'w') as f:
    f.write('{}\n'.format(len(model.state_dict().keys())))
    for i, (k, v) in enumerate(model.state_dict().items()):
        if "anchor" in k:
            continue

        vr = v.reshape(-1).cpu().numpy()
        _k = k.split('.')
        _k = _k[-2:]
        try:
            if int(_k[0]):
                _k = [_k[1]]
        except Exception as e:
            pass

        _n = '.'.join(_k)
        _name = "module_list.%i.%s"%(idx_key[k], _n)
        print(i, idx_key[k], _name, v.shape, vr.shape)
        print(v)
        input()

        f.write('{} {}'.format(
            _name,
            len(vr))    
        )

        for vv in vr:
            f.write(' ')
            f.write(struct.pack('>f' ,float(vv)).hex())
        f.write('\n')
