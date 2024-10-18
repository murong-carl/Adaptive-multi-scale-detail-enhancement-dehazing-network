
# --- Imports --- #
import time
import torch
import torch.nn.functional as F
import torchvision.utils as utils
from math import log10


def to_psnr(dehaze, gt):
    mse = F.mse_loss(dehaze, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list

def to_ssim_skimage(dehaze, gt):
    c1 = pow(0.01, 2)  
    c2 = pow(0.03, 2)  

    dehaze_list = torch.split(dehaze, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)

    ssim_list = []

    for ind in range(len(dehaze_list)):
        dehaze_img = dehaze_list[ind].squeeze()
        gt_img = gt_list[ind].squeeze()

        mu1 = torch.mean(dehaze_img)
        mu2 = torch.mean(gt_img)

        sigma1_sq = torch.var(dehaze_img)
        sigma2_sq = torch.var(gt_img)
        sigma12 = torch.mean(dehaze_img * gt_img) - mu1 * mu2

        numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
        denominator = (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2)

        ssim = numerator / denominator
        ssim_list.append(ssim.item())

    return ssim_list


def validation(net, val_data_loader, device, category, save_tag=False):
    psnr_list = []
    ssim_list = []

    for batch_id, val_data in enumerate(val_data_loader):
        with torch.no_grad():
            haze, gt, image_name = val_data
            haze = haze.to(device)
            gt = gt.to(device)
            dehaze = net(haze)

        psnr_list.extend(to_psnr(dehaze, gt))

        ssim_list.extend(to_ssim_skimage(dehaze, gt))

        if save_tag:
            save_image(dehaze, image_name, category)

    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    return avr_psnr, avr_ssim


def save_image(dehaze, image_name, category):
    dehaze_images = torch.split(dehaze, 1, dim=0)
    batch_num = len(dehaze_images)

    for ind in range(batch_num):
        utils.save_image(dehaze_images[ind], './{}_results/{}'.format(category, image_name[ind][:-3] + 'png'))


def print_log(epoch, num_epochs, one_epoch_time, train_psnr, val_psnr, val_ssim, category):
    print('({0:.0f}s) Epoch [{1}/{2}], Train_PSNR:{3:.2f}, Val_PSNR:{4:.2f}, Val_SSIM:{5:.4f}'
          .format(one_epoch_time, epoch, num_epochs, train_psnr, val_psnr, val_ssim))

    with open('./training_log/{}_log.txt'.format(category), 'a') as f:
        print('Date: {0}s, Time_Cost: {1:.0f}s, Epoch: [{2}/{3}], Train_PSNR: {4:.2f}, Val_PSNR: {5:.2f}, Val_SSIM: {6:.4f}'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                      one_epoch_time, epoch, num_epochs, train_psnr, val_psnr, val_ssim), file=f)


def adjust_learning_rate(optimizer, epoch, category, lr_decay=0.5):

    step = 20 if category == 'indoor' else 2

    if not epoch % step and epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
            print('Learning rate sets to {}.'.format(param_group['lr']))
    else:
        for param_group in optimizer.param_groups:
            print('Learning rate sets to {}.'.format(param_group['lr']))
