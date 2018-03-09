import os, argparse, glob
import numpy as np
import cv2
from scipy.misc import imread, imsave
from skimage.measure import compare_ssim
import psnr
import fastaniso
import py_utils as utils

parser = argparse.ArgumentParser()
parser.add_argument('--input_root', default='')
parser.add_argument('--dir_list',   default='dir_list.txt')
parser.add_argument('--max_images', default=-1,  type=int)
parser.add_argument('--smoothing',  default=True, action='store_false')
args = parser.parse_args()

def setupDirList():
    if args.input_root == '': 
        raise Exception('Input root not defined')
    print('[Input root]: %s' % (args.input_root))
    print('[Dir list]: %s'   % (args.dir_list))

    args.dir_list = os.path.join(args.input_root, args.dir_list)
    dir_list      = utils.readList(args.dir_list)
    if args.max_images > 0: 
        dir_list = dir_list[:args.max_images]
    return dir_list
    
def loadData(dir_name):
    flow_name = glob.glob(os.path.join(dir_name, '*.flo'))[0]
    prefix, _ = os.path.splitext(flow_name)
    in_img    = imread(prefix + '_input.jpg').astype(float)
    bg_img    = imread(prefix + '_bg.jpg').astype(float)
    mask      = imread(prefix + '_mask.png').astype(float) / 255
    rho       = imread(prefix + '_rho.png').astype(float) / 255
    flow      = utils.readFloFile(flow_name).astype(float)
    fcolor    = utils.flowToColor(flow)
    imsave(prefix + '_fcolor.jpg', fcolor)

    h, w, c = in_img.shape
    mask = np.expand_dims(mask, 2).repeat(3, 2)
    rho  = np.expand_dims(rho, 2).repeat(3, 2)
    return {'in':in_img, 'bg':bg_img, 'mask':mask, 'rho':rho, 
            'flow':flow, 'fcolor':fcolor, 'h':h, 'w': w, 'name': prefix}

def renderFinalImg(ref, warped, mask, rho):
    final = mask * (warped * rho) + (1 - mask) * ref
    return final

def warpImage(ref, flow, grid_x, grid_y):
    h, w   = grid_x.shape
    flow_x = np.clip(flow[:,:,1] + grid_x, 0, w-1)
    flow_y = np.clip(flow[:,:,0] + grid_y, 0, h-1)
    flow_x, flow_y = cv2.convertMaps(flow_x.astype(np.float32), flow_y.astype(np.float32), cv2.CV_32FC2) 
    warped_img = cv2.remap(ref, flow_x, flow_y, cv2.INTER_LINEAR)
    return warped_img

def computeError(img1, img2):
    img_psnr = psnr.psnr(img1, img2)
    gt_y     = cv2.cvtColor(cv2.cvtColor(img1.astype(np.uint8), cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2YCR_CB)[:,:,0]
    pred_y   = cv2.cvtColor(cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2YCR_CB)[:,:,0]
    img_ssim = compare_ssim(gt_y, pred_y, gaussian_weight=True)
    return img_psnr, img_ssim

def smoothingMask(mask):
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    mask = cv2.GaussianBlur(mask, (5,5), 0)
    return mask

def smoothingFlow(flow):
    flow[:,:,0] = fastaniso.anisodiff(flow[:,:,0], niter=9)
    flow[:,:,1] = fastaniso.anisodiff(flow[:,:,1], niter=9)
    return flow

def smoothingRho(rho, mask):
    rho[mask < 0.2] = 1
    rho = cv2.GaussianBlur(rho, (5,5), 0)
    return rho

def smoothingEstimation(data, grid_x, grid_y):
    smooth = {}
    smooth['mask']  = smoothingMask(data['mask'])
    smooth['rho']   = smoothingRho(data['rho'], smooth['mask'])
    smooth['flow']  = smoothingFlow(data['flow'])
    smooth['flow'][(smooth['mask'] < 0.2)[:,:,0:2]] = 0
    smooth['fcolor'] = utils.flowToColor(smooth['flow'])
    smooth['warped'] = warpImage(data['bg'], smooth['flow'], grid_x, grid_y)
    smooth['final']  = renderFinalImg(data['bg'], smooth['warped'], smooth['mask'], smooth['rho'])

    results = {}
    out = ['mask', 'rho', 'fcolor', 'final']
    for i, name in enumerate(out):
        key = '%s' % (name)
        if name in ['mask', 'rho']:
            results.update({key: smooth[name] * 255})
        else:
            results.update({key: smooth[name]})
    utils.saveResultsSeparate(data['name'] + "_smooth", results)

def evalList(dir_list):
    print('Total number of directories: %d' % len(dir_list))
    loss = {'psnr': 0, 'ssim': 0, 'psnr_bg': 0, 'ssim_bg': 0}
    for idx, dir_name in enumerate(dir_list):
        data = loadData(os.path.join(args.input_root, dir_name))
        h, w = data['h'], data['w']
        print('[%d/%d] Dir: %s, size %dx%d' % (idx, len(dir_list), dir_name, h, w))

        # Reconstructed Input Image with the estimated matte and background image
        grid_x = np.tile(np.linspace(0, w-1, w), (h, 1)).astype(float)
        grid_y = np.tile(np.linspace(0, h-1, h), (w, 1)).T.astype(float)
        data['warped'] = warpImage(data['bg'], data['flow'], grid_x, grid_y)
        data['final']  = renderFinalImg(data['bg'], data['warped'], data['mask'], data['rho'])
        imsave(data['name'] + '_final.jpg', data['final'])
        
        # Background Error
        p, s = computeError(data['bg'], data['in'])
        print('\t BG psnr: %f, ssim: %f' % (p, s))
        loss['psnr_bg'] += p; loss['ssim_bg'] += s

        # TOM-Net Error
        p, s = computeError(data['final'], data['in'])
        loss['psnr'] += p; loss['ssim'] += s
        print('\t TOMNet psnr: %f, ssim: %f' % (p, s))

        # Smoothing Environment Matte
        if args.smoothing:
            smoothingEstimation(data, grid_x, grid_y)

    print('******* Finish Testing Dir: %s\nList: %s' % (args.input_root, args.dir_list))
    with open(os.path.join(args.input_root, dir_name, 'Log'), 'w') as f:
        f.write('Input_root: %s\n' % (args.input_root))
        f.write('dir_list: %s\n'   % (args.dir_list))
        for k in loss.keys():
            print('[%s]: %f'     % (k, loss[k]/len(dir_list)))
            f.write('[%s]: %f\n' % (k, loss[k]/len(dir_list)))

if __name__ == '__main__':
    dir_list = setupDirList()
    evalList(dir_list)
