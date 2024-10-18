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


'''
# --------------------------------------------
# training code for DnCNN
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
# github: https://github.com/cszn/KAIR
#         https://github.com/cszn/DnCNN
#
# Reference:
@article{zhang2017beyond,
  title={Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising},
  author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
  journal={IEEE Transactions on Image Processing},
  volume={26},
  number={7},
  pages={3142--3155},
  year={2017},
  publisher={IEEE}
}
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
    transforms.Resize((256, 256)),  #change the size here according to the model
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize images
])


def main(json_path='KAIR-DLI/options/train_dncnn.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')

    opt = option.parse(parser.parse_args().opt, is_train=True)
    util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    opt['path']['pretrained_netG'] = init_path_G
    current_step = init_iter

    border = 0
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
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
    logger.info('Random seed: {}'.format(seed))
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

    train_file = "KAIR-DLI/datasets-dli/dataloader_train_revisedv2.pkl"
    with open(train_file, 'rb') as f:
        train_loader = pickle.load(f)

    test_file = "KAIR-DLI/datasets-dli/dataloader_val_revisedv2.pkl"
    with open(test_file, 'rb') as f:
        test_loader = pickle.load(f)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)

    if opt['merge_bn'] and current_step > opt['merge_bn_startpoint']:
        logger.info('^_^ -----merging bnorm----- ^_^')
        model.merge_bnorm_test()

    logger.info(model.info_network())
    model.init_train()
    logger.info(model.info_params())

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''

    for epoch in range(1000000):  # keep running
        for i, train_data in enumerate(train_loader):

            current_step += 1
            #import pdb;pdb.set_trace()

           

            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)

            # -------------------------------
            # 2) feed patch pairs>>> t = torch.from_numpy(a)
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
            # merge bnorm
            # -------------------------------
            if opt['merge_bn'] and opt['merge_bn_startpoint'] == current_step:
                logger.info('^_^ -----merging bnorm----- ^_^')
                model.merge_bnorm_train()
                model.print_network()

            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

            # -------------------------------
            # 6) testing
            # -------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0:

                avg_psnr = 0.0
                idx = 0

                for test_data in test_loader:
                    idx += 1
                    image_name_ext = os.path.basename(test_data['L_path'][0])
                    img_name, ext = os.path.splitext(image_name_ext)

                    img_dir = os.path.join(opt['path']['images'], img_name)
                    util.mkdir(img_dir)

                    model.feed_data(test_data)
                    model.test()

                    visuals = model.current_visuals()
                    E_img = util.tensor2uint(visuals['E'])
                    H_img = util.tensor2uint(visuals['H'])

                    # -----------------------
                    # save estimated image E
                    # -----------------------
                    save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
                    util.imsave(E_img, save_img_path)

                    # -----------------------
                    # calculate PSNR
                    # -----------------------
                    current_psnr = util.calculate_psnr(E_img, H_img, border=border)

                    logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB'.format(idx, image_name_ext, current_psnr))

                    avg_psnr += current_psnr

                avg_psnr = avg_psnr / idx

                # testing log
                logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB\n'.format(epoch, current_step, avg_psnr))

    logger.info('Saving the final model.')
    model.save('latest')
    logger.info('End of training.')


if __name__ == '__main__':
    main()
