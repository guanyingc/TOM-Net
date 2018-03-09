import numpy as np
import os, datetime
import matplotlib
from scipy.misc import imsave

def makeFile(f):
    if not os.path.exists(f):
        os.makedirs(f)

def getDateTime(minutes=False):
    t = datetime.datetime.now()
    dt = ('%s_%s' % (t.month, t.day))
    if minutes:
        dt += '_%s-%s-%s' % (t.hour, t.minute, t.second)
    return dt

def checkEmpty(a_list):
    if len(a_list) == 0:
        raise Exception('Empty list')

def readList(list_path,ignore_head=False):
    lists = []
    with open(list_path) as f:
        lists = f.read().splitlines()
    if ignore_head:
        lists = lists[1:]
    return lists

def readFloFile(filename, short=True):
    with open(filename, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if magic != 202021.25:
            raise Exception('Magic number incorrect: %s' % filename)
        w = int(np.fromfile(f, np.int32, count=1))
        h = int(np.fromfile(f, np.int32, count=1))
        
        if short:
            flow = np.fromfile(f, np.int16, count=h*w*2)
            flow = flow.astype(np.float32)
        else:
            flow = np.fromfile(f, np.float32, count=h * w * 2)
        flow = flow.reshape((h, w, 2))
    return flow

def flowToMap(F_mag, F_dir):
    sz = F_mag.shape
    flow_color = np.zeros((sz[0], sz[1], 3), dtype=float)
    flow_color[:,:,0] = (F_dir+np.pi) / (2 * np.pi)
    f_dir =(F_dir+np.pi) / (2 * np.pi)
    flow_color[:,:,1] = F_mag / 255 #F_mag.max()
    flow_color[:,:,2] = 1
    flow_color = matplotlib.colors.hsv_to_rgb(flow_color)*255
    return flow_color

def flowToColor(flow):
    F_dx = flow[:,:,1].copy().astype(float)
    F_dy = flow[:,:,0].copy().astype(float)
    F_mag = np.sqrt(np.power(F_dx, 2) + np.power(F_dy, 2))
    F_dir = np.arctan2(F_dy, F_dx)
    flow_color = flowToMap(F_mag, F_dir)
    return flow_color.astype(np.uint8)

def saveResultsSeparate(prefix, results):
    for key in results:
        save_name = prefix + '_' + key
        value = results[key]
        if key == 'mask' or key == 'rho':
            save_name += '.png'
        else:
            save_name += '.jpg'
        imsave(save_name, value)

