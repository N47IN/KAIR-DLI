import os.path
import math
import pickle
import argparse
import random
import numpy as np
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
import os.path
import math
import argparse
import time
import random
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pickle

import pickle
from collections import OrderedDict
from torch.utils.data import Dataset

import logging
import torch
from torch.utils.data import DataLoader


from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option

from data.select_dataset import define_Dataset
from models.select_model import define_Model


from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model


'''
# --------------------------------------------
# training code for MSRResNet
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
# github: https://github.com/cszn/KAIR
# --------------------------------------------
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''

class DefectDataset(Dataset):
    def __init__(self, root_dir,load_type, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.triplets = self._load_triplets(load_type)

    def _load_triplets(self,load_type):
        triplets = []
        for object_name in os.listdir(self.root_dir):
            #print(object_name)
            object_path = os.path.join(self.root_dir, object_name, load_type)
            degraded_path = os.path.join(object_path, 'Degraded_image')
            mask_path = os.path.join(object_path, 'Defect_mask')
            clean_path = os.path.join(object_path, 'GT_clean_image')
            # Iterate through each defect type (broken_large, broken_small, etc.)
            for defect_type in os.listdir(degraded_path):
                #print(defect_type)
                degraded_defect_dir = os.path.join(degraded_path, defect_type)
                mask_defect_dir = os.path.join(mask_path, defect_type)
                clean_defect_dir = os.path.join(clean_path, defect_type)
                # Match image triplets by name in each defect folder
                for img_name in os.listdir(degraded_defect_dir):
                    degraded_img = os.path.join(degraded_defect_dir, img_name)
                    mask_img = os.path.join(mask_defect_dir, img_name.replace(".png", "_mask.png"))
                    clean_img = os.path.join(clean_defect_dir, img_name)
                    # Only add the triplet if all three files exist
                    if os.path.exists(degraded_img) and os.path.exists(mask_img) and os.path.exists(clean_img):
                        triplets.append((degraded_img,clean_img))
        return triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        degraded_img_path, clean_img_path = self.triplets[idx]
        degraded_img = Image.open(degraded_img_path).convert("RGB")
        clean_img = Image.open(clean_img_path).convert("RGB")

        # Apply transforms, if any
        if self.transform:
            degraded_img = self.transform(degraded_img)
            # mask_img = self.transform(mask_img)
            clean_img = self.transform(clean_img)

        return degraded_img, clean_img
    
transform = transforms.Compose([
    transforms.Resize((900, 900)),  #change the size here according to the model
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),    # Normalize images
])

def main(json_path='options/train_msrresnet_psnr.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    border = opt['scale']
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    train_file = "/home/navin/coursework/kla-challenge-dli/KAIR-DLI/datasets-dli/dataloader_train_revisedv3.pkl"
    with open(train_file, 'rb') as f:
        train_loader = pickle.load(f)
        print("_________________")

    test_file = "/home/navin/coursework/kla-challenge-dli/KAIR-DLI/datasets-dli/dataloader_val_revisedv3.pkl"
    with open(test_file, 'rb') as f:
        test_loader = pickle.load(f)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)
    model.init_train()
    if opt['rank'] == 0:
        logger.info(model.info_network())
        logger.info(model.info_params())

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''

    for epoch in range(1000000):  # keep running
        

        for i, train_data in enumerate(train_loader):

            current_step += 1

            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)

            # -------------------------------
            # 2) feed patch pairs
            degraded_imgs, clean_imgs = train_data
            degraded_imgs = degraded_imgs.to("cuda")
            clean_imgs = clean_imgs.to("cuda")
            # -------------------------------
            model.feed_data(degraded_imgs, clean_imgs)

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step)

            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

            # -------------------------------
            # 6) testing
            # -------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:

                avg_psnr = 0.0
                idx = 0

                for test_data in test_loader:
                    idx += 1
                    degraded_imgs, clean_imgs = test_data
                    degraded_imgs = degraded_imgs.to("cuda")
                    clean_imgs = clean_imgs.to("cuda")
                    img_dir = opt['path']['images']
                    util.mkdir(img_dir)

                    model.feed_data(degraded_imgs, clean_imgs)
                    model.test()

                    visuals = model.current_visuals()
                    E_img = util.tensor2uint(visuals['E'])
                    H_img = util.tensor2uint(visuals['H'])

                    # -----------------------
                    # save estimated image E
                    # -----------------------
                    save_img_path = os.path.join(img_dir, '{:d}_{:d}.png'.format(idx, current_step))
                    util.imsave(E_img, save_img_path)

                    # -----------------------
                    # calculate PSNR
                    # -----------------------
                    current_psnr = util.calculate_psnr(E_img, H_img, border=border)

                    logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB'.format(idx, save_img_path, current_psnr))

                    avg_psnr += current_psnr

                avg_psnr = avg_psnr / idx

                # testing log
                logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB\n'.format(epoch, current_step, avg_psnr))

if __name__ == '__main__':
    main()
