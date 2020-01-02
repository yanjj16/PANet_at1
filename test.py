"""Evaluation Script"""
import os
import shutil
import tqdm
import numpy as np
import torch
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torchvision.transforms import Compose
import torchvision.transforms.functional as tr_F
from PIL import Image

from models.fewshot import FewShotSeg
from dataloaders.test import davis2017_test
from dataloaders.transforms import ToTensorNormalize
from dataloaders.transforms import Resize
from util.utils import set_seed, CLASS_LABELS, get_bbox

config = {'model': {'align': False},
          'dataset': 'davis',
          'batch_size': 1,
          'train':True,
          'seed': 1234,
          'gpu_id':0,
          'n_runs':1,
          'cuda_visable': '0',
          'input_size': (321, 321),
          'snapshots':'./snapshots/',
          'task': {'n_ways': 1,
                   'n_shots': 1,
                   'n_queries': 1},
          'path': {'davis': {'data_dir': '../Dataset/DAVIS2017/DAVIS',
                             'data_split': 'train'}}
          }

def get_fg_mask(label, class_id):
    """
    Generate FG/BG mask from the segmentation mask

    Args:
        label:
            semantic mask
        scribble:
            scribble mask
        class_id:
            semantic class of interest
        class_ids:
            all class id in this episode
    """
    # Dense Mask

    fg_mask = torch.where(label == class_id,
                          torch.ones_like(label), torch.zeros_like(label))
    return {'fg_mask': fg_mask}

def get_bg_mask(label,class_ids):
    bg_mask = torch.ones_like(label)
    for class_id in class_ids:
        bg_mask[label == class_id] = 0
    return  {'bg_mask': bg_mask}

def process_test_data(sample,label,class_ids):
    pass

def main(_run, config):

    palette_path = config['palette_dir']
    with open(palette_path) as f:
        palette = f.readlines()
    palette = list(np.asarray([[int(p) for p in pal[0:-1].split(' ')] for pal in palette]).reshape(768))

    n_shots = config['task']['n_shots']
    n_ways = config['task']['n_ways']
    set_seed(config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=config['gpu_id'])
    torch.set_num_threads(1)
    model = FewShotSeg( cfg=config['model'])
    model = nn.DataParallel(model.cuda(), device_ids=[config['gpu_id'],])
    if config['train']:
        model.load_state_dict(torch.load(config['snapshot'], map_location='cpu'))
    model.eval()

    data_name = config['dataset']
    if data_name == 'davis':
        make_data = davis2017_test
    else:
        raise ValueError('Wrong config for dataset!')
    labels = CLASS_LABELS[data_name]['val']
    transforms = [Resize(size=config['input_size'])]
    transforms = Compose(transforms)

    with torch.no_grad():
        for run in range(config['n_runs']):
            set_seed(config['seed'] + run)
            dataset = make_data(
                base_dir=config['path'][data_name]['data_dir'],
                split=config['path'][data_name]['data_split'],
                transforms=transforms,
                to_tensor=ToTensorNormalize(),
                labels=labels
            )

            testloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False,
                                    num_workers=1, pin_memory=True, drop_last=False)

            for iteration, batch in enumerate(testloader):
                class_name = batch[2]
                num_frame = batch[1]
                all_class_data = batch[0]
                class_ids = all_class_data[0]['obj_ids']
                support_images = [[all_class_data[0]['image'].cuda() for _ in range(n_shots)]
                                  for _ in range(n_ways)]
                support_mask = all_class_data[0]['label'][labels[iteration]]
                support_fg_mask = [[get_fg_mask(support_mask, class_ids[way])
                                    for shot in range(n_shots)] for way in range(len(class_ids))]
                support_bg_mask = [[get_bg_mask(support_mask,class_ids)
                        for _ in range(n_shots)] for _ in range(n_ways)]
                s_fg_mask = [[shot['fg_mask'].float().cuda() for shot in way]
                             for way in support_fg_mask]
                s_bg_mask = [[shot['bg_mask'].float().cuda() for shot in way]
                             for way in support_bg_mask]

                for idx, data in enumerate(all_class_data):
                    query_images = [all_class_data[idx]['image'].cuda() for i in range(n_ways)]
                    query_labels = torch.cat([query_label.cuda() for query_label
                                              in [all_class_data[idx]['label'][labels[iteration]],]])

                    if idx > 0:
                        pre_mask = [pred_mask,]
                    elif idx ==0 :
                        pre_mask = [support_mask,]
                    query_pred, _ = model(support_images, s_fg_mask, s_bg_mask, query_images, pre_mask)
                    pred = query_pred.argmax(dim=1, keepdim=True)
                    pred = pred.data.cpu().numpy()
                    img = pred[0, 0]
                    for i in range(img.shape[0]):
                        for j in range(img.shape[1]):
                            if img[i][j] > 0:
                                print(f'{img[i][j]} {len(support_fg_mask)}')

                    img_e = Image.fromarray(img.astype('float32')).convert('P')
                    pred_mask = tr_F.resize(img_e, config['input_size'], interpolation=Image.NEAREST)
                    pred_mask = torch.Tensor(np.array(pred_mask))
                    pred_mask = torch.unsqueeze(pred_mask,dim=0)
                    img_e.putpalette(palette)
                    img_e.save(os.path.join('./result/',f'{class_name}','/', '{:05d}.png'.format(idx)))



