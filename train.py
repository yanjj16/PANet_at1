"""Training Script"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
from torchvision.transforms import Compose


from models.fewshot import FewShotSeg
from dataloaders.transforms import RandomMirror, Resize, ToTensorNormalize
from dataloaders.davis2017few import davis2017_fewshot
from util.utils import set_seed, CLASS_LABELS

config = {'model': {'align': True},
          'dataset': 'davis',
          'n_steps': 120000,
          'batch_size': 1,
          'seed': 1234,
          'lr_milestones': [45000, 90000, 110000],
          'align_loss_scaler': 1,
          'ignore_label': 255,
          'print_interval': 100,
          'save_pred_every': 5000,
          'cuda_visable': '0',
          'label_sets': 0,
          'input_size': (448, 448),
          'snapshots':'./snapshots/',
          'task': {'n_ways': 1,
                   'n_shots': 1,
                   'n_queries': 1},
          'optim': {'lr': 1e-3,
                    'momentum': 0.9,
                    'weight_decay': 0.0005},
          'log_dir': './runs',
          'path': {'davis': {'data_dir': '../Dataset/DAVIS2017/DAVIS',
                             'data_split': 'train'}},
          'base_dir': ''
          }


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="PANet Network")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    return parser.parse_args()


args = get_arguments()


def main(config):
    if not os.path.exists(config['snapshots']):
        os.makedirs(config['snapshots'])
    snap_shots_dir = config['snapshots']
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.cuda.set_device(2)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    set_seed(config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = True

    torch.set_num_threads(1)

    model = FewShotSeg(cfg=config['model'])
    model = nn.DataParallel(model.cuda(),device_ids=[2])
    model.train()

    data_name = config['dataset']
    if data_name == 'davis':
        make_data = davis2017_fewshot
    else:
        raise ValueError('Wrong config for dataset!')
    labels = CLASS_LABELS[data_name][config['label_sets']]
    transforms = Compose([Resize(size=config['input_size']),
                          RandomMirror()])
    dataset = make_data(
        base_dir=config['path'][data_name]['data_dir'],
        split=config['path'][data_name]['data_split'],
        transforms=transforms,
        to_tensor=ToTensorNormalize(),
        labels=labels,
        max_iters=config['n_steps'] * config['batch_size'],
        n_ways=config['task']['n_ways'],
        n_shots=config['task']['n_shots'],
        n_queries=config['task']['n_queries']
    )
    trainloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )

    optimizer = torch.optim.SGD(model.parameters(), **config['optim'])
    scheduler = MultiStepLR(optimizer, milestones=config['lr_milestones'], gamma=0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=config['ignore_label'])

    i_iter = 0
    log_loss = {'loss': 0, 'align_loss': 0}

    for i_iter, sample_batched in enumerate(trainloader):
        # Prepare input
        support_images = [[shot.cuda() for shot in way]
                          for way in sample_batched['support_images']]
        support_fg_mask = [[shot[f'fg_mask'].float().cuda() for shot in way]
                           for way in sample_batched['support_fg_mask']]
        support_bg_mask = [[shot[f'bg_mask'].float().cuda() for shot in way]
                           for way in sample_batched['support_bg_mask']]
        query_images = [query_image.cuda()
                        for query_image in sample_batched['query_images']]
        query_labels = torch.cat(
            [query_label.long().cuda() for query_label in sample_batched['query_labels']], dim=0)
        # print(query_labels.shape)
        pre_masks = [query_label.float().cuda() for query_label in sample_batched['query_masks']]
        # Forward and Backward
        optimizer.zero_grad()
        query_pred, align_loss = model(support_images, support_fg_mask, support_bg_mask,
                                       query_images,pre_masks)
        query_loss = criterion(query_pred, query_labels)
        loss = query_loss + align_loss * config['align_loss_scaler']

        loss.backward()
        optimizer.step()
        scheduler.step()

        # Log loss
        query_loss = query_loss.detach().data.cpu().numpy()
        align_loss = align_loss.detach().data.cpu().numpy() if align_loss != 0 else 0
        # _run.log_scalar('loss', query_loss)
        # _run.log_scalar('align_loss', align_loss)
        log_loss['loss'] += query_loss
        log_loss['align_loss'] += align_loss

        # print loss and take snapshots
        if (i_iter + 1) % config['print_interval'] == 0:
            loss = log_loss['loss'] / (i_iter + 1)
            align_loss = log_loss['align_loss'] / (i_iter + 1)
            print(f'step {i_iter + 1}: loss: {loss}, align_loss: {align_loss}')

        if (i_iter + 1) % config['save_pred_every'] == 0:
            torch.save(model.state_dict(),
                       os.path.join(f'{snap_shots_dir}', f'{i_iter + 1}.pth'))

    torch.save(model.state_dict(),
               os.path.join(f'{snap_shots_dir}', f'{i_iter + 1}.pth'))


if __name__ == '__main__':
    main(config=config)