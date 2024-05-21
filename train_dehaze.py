import argparse
import datetime
import os
from tqdm import tqdm
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader
from data.loader import PairLoader, SingleLoader, dataset_collate
from model.dehaze.network import create_model
from model.detect.yolo import YoloBody
from model.detect.yolo_test import YOLO
from model.detect.yolo_train import YOLOLoss, get_lr_scheduler, set_optimizer_lr
from utils import write_img, chw_to_hwc, pad_img
from utils.callbacks import LossHistory
from utils.utils import get_anchors, get_classes, get_lr
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--detect_model_path', default='logs/ep300-loss0.038-val_loss0.040.pth', type=str)
args = parser.parse_args()


def get_state_dict(model_path):
    state_dict = torch.load(model_path)['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    return new_state_dict


def test(test_loader, network, yolo):
    print(f'input_dir: {args.input_dir}')
    print(f'output_dir: {args.output_dir}')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    torch.cuda.empty_cache()
    network.eval()
        
    with tqdm(total=len(test_loader), desc='Dehazing') as pbar:
        for idx, batch in enumerate(test_loader):
            input = batch['hazy'].cuda()
            filename = batch['filename'][0]

            with torch.no_grad():
                H, W = input.shape[2:]
                input = pad_img(input, network.patch_size if hasattr(network, 'patch_size') else 16)
                guidance = yolo.get_detection_guidance(input, resize=True).cuda()
                output = network(input, guidance).clamp_(-1, 1)
                output = output[:, :, :H, :W]
                # [-1, 1] to [0, 1]
                output = output * 0.5 + 0.5

            out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
            write_img(os.path.join(args.output_dir, filename), out_img)
            pbar.update(1)



def fit_one_epoch(network, yolo, yolo_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, train_dataloader, test_dataloader, UnFreeze_Epoch, Cuda, save_period, save_dir):
    loss        = 0
    val_loss    = 0
    Dehazy_loss = 0
    criterion = torch.nn.MSELoss()

    if Cuda:
        network = network.cuda()
        criterion = criterion.cuda()

    network.train()
    print('Start Train')
    
    with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{UnFreeze_Epoch}', postfix=dict, mininterval=0.3) as pbar:
        
        for iteration, batch in enumerate(train_dataloader):
            images, clear_images, targets = batch['hazy'], batch['clear'], batch['detect_label']
            if Cuda:
                images  = images.cuda()
                clear_images = clear_images.cuda()
                targets = targets.cuda()
            guidance = yolo.get_detection_guidance(images, resize=True)
            optimizer.zero_grad()
            images = pad_img(images, network.patch_size if hasattr(network, 'patch_size') else 16)
            
            if Cuda:
                guidance = guidance.cuda()
            outputs = network(images, guidance)
            outputs = outputs[:, :, :clear_images.shape[2], :clear_images.shape[3]]

            loss_dehazy = criterion(outputs, clear_images)
            outputs_det = outputs.clamp(-1, 1) * 0.5 + 0.5
            yolo_det = yolo(outputs_det)
            # if Cuda:
            #     yolo_det = yolo_det.cuda()
            loss_yolo = yolo_loss(yolo_det, targets, outputs_det)
            if Cuda:
                loss_yolo = loss_yolo.cuda()
            loss_value = loss_dehazy + 0.4 * loss_yolo

            # print("This is dehazy loss", loss_dehazy.device)
            # print("This is yolo loss", loss_yolo.device)
            # print("This is total loss", loss_value.device)

            loss_value.backward()
            optimizer.step()

            loss += loss_value.item()
            Dehazy_loss += loss_dehazy.item()

            pbar.set_postfix(**{'loss'  : loss / (iteration + 1),
                                'Dehazy_loss': Dehazy_loss / (iteration + 1),
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)
    print('Finish Train')
    save_state_dict = network.state_dict()
    torch.save(save_state_dict, os.path.join(save_dir, f'FriendNet_epoch_{epoch + 1}.pth'))


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "5, 6"
    network = create_model()
    # network.load_state_dict(torch.load('logs_dehazed/FriendNet_epoch_7.pth'))
    network.train()
    network = network.cuda()

    Cuda = True

    Init_Epoch          = 0
    Freeze_Epoch        = 0
    Freeze_batch_size   = 16

    UnFreeze_Epoch      = 100
    Unfreeze_batch_size = 16

    Freeze_Train        = False
    

    Init_lr             = 1e-2
    Min_lr              = Init_lr * 0.01

    optimizer_type      = "adamw"
    momentum            = 0.937
    weight_decay        = 5e-4

    lr_decay_type       = "cos"

    save_period         = 5
    save_dir            = 'logs_dehazed'

    eval_flag           = True
    eval_period         = 10

    num_workers         = 0

    batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
    nbs             = 64
    lr_limit_max    = 1e-3 if optimizer_type == 'adamw' else 5e-2
    lr_limit_min    = 3e-4 if optimizer_type == 'adamw' else 5e-4
    Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    classes_path    = 'data/voc_classes.txt'
    anchors_path    = 'data/yolo_anchors.txt'
    anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    input_shape     = [256, 256]
    label_smoothing     = 0

    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors     = get_anchors(anchors_path)

    yolo_loss    = YOLOLoss(anchors, num_classes, input_shape, anchors_mask, label_smoothing)
    time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(save_dir, 'loss_' + str(time_str))
    loss_history = LossHistory(log_dir, network, input_shape=input_shape)

    yolo = YoloBody(anchors_mask, num_classes, pretrained=False)
    model_dict      = yolo.state_dict()
    pretrained_dict = torch.load(args.detect_model_path, map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    model_dict.update(temp_dict)
    yolo.load_state_dict(model_dict)
    yolo.cuda()
    yolo.eval()

    optimizer = torch.optim.AdamW(network.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay)

    train_dataset = PairLoader()
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True, collate_fn=dataset_collate)
    num_train = len(train_dataset)
    test_dataset = PairLoader(mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=0, pin_memory=True, collate_fn=dataset_collate)
    num_val = len(test_dataset)
    # test(test_loader, network, yolo)
    # train(train_dataloader, network, yolo)

    epoch_step      = num_train // batch_size
    epoch_step_val  = num_val // batch_size
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
    UnFreeze_flag = False

    for epoch in range(Init_Epoch, UnFreeze_Epoch):

        if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
            batch_size = Unfreeze_batch_size

            nbs         = 64
            Init_lr_fit = max(batch_size / nbs * Init_lr, 1e-4)
            Min_lr_fit  = max(batch_size / nbs * Min_lr, 1e-6)

            lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

            epoch_step      = num_train // batch_size
            epoch_step_val  = num_val // batch_size

            if epoch_step == 0 or epoch_step_val == 0:
                raise ValueError("Dataset error!")

            UnFreeze_flag = True

        train_dataloader.dataset.epoch_now  = epoch
        test_dataloader.dataset.epoch_now   = epoch

        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        fit_one_epoch(network, yolo, yolo_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, train_dataloader, test_dataloader, UnFreeze_Epoch, Cuda, save_period, save_dir)


if __name__ == '__main__':
    main()