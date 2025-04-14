import numpy as np
import cv2
import os
import os.path as ops
import copy
import math
import ujson as json
from scipy.interpolate import interp1d
import matplotlib
from tqdm import tqdm
import warnings
import pickle
import pdb

import mmcv
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .utils import *
from .MinCostFlow import SolveMinCostFlow

plt.rcParams['figure.figsize'] = (35, 30)
plt.rcParams.update({'font.size': 25})
plt.rcParams.update({'font.weight': 'semibold'})

color = [[0, 0, 255],  # red
         [0, 255, 0],  # green
         [255, 0, 255],  # purple
         [255, 255, 0]]  # cyan

vis_min_y = 5
vis_max_y = 80


def parse_line(line):
    """解析单行JSON"""
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        print(f"解析失败的行: {line.strip()}")
        return None
        
class LaneVis(object):
    def __init__(self, db):
        self.resize_h = db.resize_h
        self.resize_w = db.resize_w
        self.H_crop = homography_crop_resize([db.org_h, db.org_w], db.crop_y, [db.resize_h, db.resize_w])
        self.top_view_region = db.top_view_region
        self.ipm_h = db.ipm_h
        self.ipm_w = db.ipm_w
        self.org_h = db.org_h
        self.org_w = db.org_w
        self.crop_y = db.crop_y
        self.x_min = db.top_view_region[0, 0]
        self.x_max = db.top_view_region[1, 0]
        self.y_min = db.top_view_region[2, 1]
        self.y_max = db.top_view_region[0, 1]
        self.y_samples = np.array([  5,  10,  15,  20,  30,  40,  50,  60,  80,  100], dtype=np.float32)
        self.dist_th = 1.5
        self.ratio_th = 0.75
        self.close_range = 40
        self.num_category = 21
        self.category_dict = {0: 'invalid',
                              1: 'white-dash',
                              2: 'white-solid',
                              3: 'double-white-dash',
                              4: 'double-white-solid',
                              5: 'white-ldash-rsolid',
                              6: 'white-lsolid-rdash',
                              7: 'yellow-dash',
                              8: 'yellow-solid',
                              9: 'double-yellow-dash',
                              10: 'double-yellow-solid',
                              11: 'yellow-ldash-rsolid',
                              12: 'yellow-lsolid-rdash',
                              13: 'fishbone',
                              14: 'others',
                              20: 'roadedge'}

    def vis(self, gt, pred, save_dir, img_dir, img_name, prob_th=0.5):
        img_path = os.path.join(img_dir, 'images', img_name)
        fig = plt.figure()
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133, projection='3d')

        pred_lanelines = pred['lane_lines']
        pred_lanes = [np.array(lane['xyz']) for i, lane in enumerate(pred_lanelines)]
        pred_category = [int(lane['category']) for i, lane in enumerate(pred_lanelines)]
        pred_prob = [float(lane['laneLines_prob']) for i, lane in enumerate(pred_lanelines)]
        pred_lanes = [pred_lanes[ii] for ii in range(len(pred_prob)) if
                            pred_prob[ii] > prob_th]
        pred_prob = [pred_prob[ii] for ii in range(len(pred_prob)) if
                            pred_prob[ii] > prob_th]

        # evaluate lanelines
        cam_extrinsics = np.array(gt['extrinsic'])
        # Re-calculate extrinsic matrix based on ground coordinate
        R_vg = np.array([[0, 1, 0],
                            [-1, 0, 0],
                            [0, 0, 1]], dtype=float)
        R_gc = np.array([[1, 0, 0],
                            [0, 0, 1],
                            [0, -1, 0]], dtype=float)
        cam_extrinsics[:3, :3] = np.matmul(np.matmul(
                                    np.matmul(np.linalg.inv(R_vg), cam_extrinsics[:3, :3]),
                                        R_vg), R_gc)

        cam_extrinsics[0:2, 3] = 0.0

        cam_intrinsics = gt['intrinsic']
        cam_intrinsics = np.array(cam_intrinsics)
        gt_lanes_packed = gt['lane_lines']

        gt_lanes, gt_visibility, gt_category = [], [], []
        for j, gt_lane_packed in enumerate(gt_lanes_packed):
            # A GT lane can be either 2D or 3D
            # if a GT lane is 3D, the height is intact from 3D GT, so keep it intact here too
            lane = np.array(gt_lane_packed['xyz'])
            lane_visibility = np.ones(len(lane[0]))

            lane = np.vstack((lane, np.ones((1, lane.shape[1]))))
            cam_representation = np.linalg.inv(
                                    np.array([[0, 0, 1, 0],
                                                [-1, 0, 0, 0],
                                                [0, -1, 0, 0],
                                                [0, 0, 0, 1]], dtype=float))

            lane = np.matmul(cam_extrinsics, np.matmul(cam_representation, lane))
            lane = lane[0:3, :].T

            gt_lanes.append(lane)
            gt_visibility.append(lane_visibility)
            gt_category.append(gt_lane_packed['category'])

        pred_category = [pred_category[k] for k, lane in enumerate(pred_lanes)
                        if lane[0, 1] < self.y_samples[-1] and lane[-1, 1] > self.y_samples[0]]
        pred_prob = [pred_prob[k] for k, lane in enumerate(pred_lanes)
                        if lane[0, 1] < self.y_samples[-1] and lane[-1, 1] > self.y_samples[0]]
        pred_lanes = [lane for lane in pred_lanes if lane[0, 1] < self.y_samples[-1] and lane[-1, 1] > self.y_samples[0]]

        pred_lanes = [prune_3d_lane_by_range(np.array(lane), self.x_min, self.x_max) for lane in pred_lanes]

        pred_category = [pred_category[k] for k, lane in enumerate(pred_lanes) if lane.shape[0] > 1]
        pred_prob = [pred_prob[k] for k, lane in enumerate(pred_lanes) if lane.shape[0] > 1]
        pred_lanes = [lane for lane in pred_lanes if lane.shape[0] > 1]

        # only consider those gt lanes overlapping with sampling range
        gt_category = [gt_category[k] for k, lane in enumerate(gt_lanes)
                        if lane[0, 1] < self.y_samples[-1] and lane[-1, 1] > self.y_samples[0]]
        gt_lanes = [lane for lane in gt_lanes if lane[0, 1] < self.y_samples[-1] and lane[-1, 1] > self.y_samples[0]]

        gt_lanes = [prune_3d_lane_by_range(np.array(lane), self.x_min, self.x_max) for lane in gt_lanes]

        gt_category = [gt_category[k] for k, lane in enumerate(gt_lanes) if lane.shape[0] > 1]
        gt_lanes = [lane for lane in gt_lanes if lane.shape[0] > 1]

        cnt_gt = len(gt_lanes)
        cnt_pred = len(pred_lanes)

        P_g2im = projection_g2im_extrinsic(cam_extrinsics, cam_intrinsics)
        P_gt = np.matmul(self.H_crop, P_g2im)
        img = cv2.imread(img_path)
        img = cv2.warpPerspective(img, self.H_crop, (self.resize_w, self.resize_h))
        img = img.astype(np.float) / 255
        
        H_ipm2g = cv2.getPerspectiveTransform(
            np.float32([[0, 0], [self.ipm_w - 1, 0], [0, self.ipm_h - 1], [self.ipm_w - 1, self.ipm_h - 1]]), 
            np.float32(self.top_view_region))
        H_g2ipm = np.linalg.inv(H_ipm2g)
        H_g2im = homography_g2im_extrinsic(cam_extrinsics, cam_intrinsics)
        H_im2g = np.linalg.inv(H_g2im)
        H_im2ipm = np.linalg.inv(np.matmul(H_g2im, H_ipm2g))
        raw_img = cv2.imread(img_path)
        raw_img = raw_img.astype(np.float) / 255
        im_ipm = cv2.warpPerspective(raw_img, H_im2ipm, (self.ipm_w, self.ipm_h))
        im_ipm = np.clip(im_ipm, 0, 1)

        ax1.imshow(img[:, :, [2, 1, 0]])
        ax2.imshow(im_ipm[:, :, [2, 1, 0]])

        gt_visibility_mat = np.zeros((cnt_gt, 10))
        pred_visibility_mat = np.zeros((cnt_pred, 10))

        for i in range(cnt_pred):
            # # ATTENTION: ensure y mono increase before interpolation: but it can reduce size
            # pred_lanes[i] = make_lane_y_mono_inc(np.array(pred_lanes[i]))
            # pred_lane = prune_3d_lane_by_range(np.array(pred_lanes[i]), self.x_min, self.x_max)
            min_y = np.min(np.array(pred_lanes[i])[:, 1])
            max_y = np.max(np.array(pred_lanes[i])[:, 1])
            x_values, z_values, visibility_vec = resample_laneline_in_y(np.array(pred_lanes[i]), self.y_samples,
                                                                        out_vis=True)
            pred_lanes[i] = np.vstack([x_values, z_values]).T
            pred_visibility_mat[i, :] = np.logical_and(x_values >= self.x_min,
                                                       np.logical_and(x_values <= self.x_max,
                                                                      np.logical_and(self.y_samples >= min_y,
                                                                                     self.y_samples <= max_y)))
            pred_visibility_mat[i, :] = np.logical_and(pred_visibility_mat[i, :], visibility_vec)

        for i in range(cnt_gt):
            min_y = np.min(np.array(gt_lanes[i])[:, 1])
            max_y = np.max(np.array(gt_lanes[i])[:, 1])
            x_values, z_values, visibility_vec = resample_laneline_in_y(np.array(gt_lanes[i]), self.y_samples,
                                                                        out_vis=True)
            gt_lanes[i] = np.vstack([x_values, z_values]).T
            gt_visibility_mat[i, :] = np.logical_and(x_values >= self.x_min,
                                                     np.logical_and(x_values <= self.x_max,
                                                                    np.logical_and(self.y_samples >= min_y,
                                                                                   self.y_samples <= max_y)))
            gt_visibility_mat[i, :] = np.logical_and(gt_visibility_mat[i, :], visibility_vec)

        for i in range(cnt_pred):
            x_values = pred_lanes[i][:, 0]
            z_values = pred_lanes[i][:, 1]
            lane_cate = pred_category[i]
            x_g = x_values[np.where(pred_visibility_mat[i, :])]
            y_g = self.y_samples[np.where(pred_visibility_mat[i, :])]
            z_g = z_values[np.where(pred_visibility_mat[i, :])]
            if len(x_g) == 0:
                continue

            fit1 = np.polyfit(y_g, x_g, 2)
            fit2 = np.polyfit(y_g, z_g, 2)
            f_xy = np.poly1d(fit1)
            f_zy = np.poly1d(fit2)
            y_g = np.linspace(min(y_g), max(y_g), 5 * len(y_g))
            x_g = f_xy(y_g)
            z_g = f_zy(y_g)
            x_2d, y_2d = projective_transformation(P_gt, x_g, y_g, z_g)
            valid_mask_2d = np.logical_and(np.logical_and(x_2d >= 0, x_2d < self.resize_w), np.logical_and(y_2d >= 0, y_2d < self.resize_h))
            x_2d = x_2d[valid_mask_2d]
            y_2d = y_2d[valid_mask_2d]

            x_ipm, y_ipm = homographic_transformation(H_g2ipm, x_g, y_g)
            valid_mask_ipm = np.logical_and(np.logical_and(x_ipm >= 0, x_ipm < self.ipm_w), np.logical_and(y_ipm >= 0, y_ipm < self.ipm_h))
            x_ipm = x_ipm[valid_mask_ipm]
            y_ipm = y_ipm[valid_mask_ipm]

            if lane_cate == 1: # white dash
                ax1.plot(x_2d, y_2d, 'mediumpurple', lw=3, alpha=0.8, label='pred')
                ax1.plot(x_2d, y_2d, 'white', lw=1, alpha=0.8, linestyle=(0,(20,10)))
                ax2.plot(x_ipm, y_ipm, 'mediumpurple', lw=3, alpha=0.8, label='pred')
                ax2.plot(x_ipm, y_ipm, 'white', lw=1, alpha=0.8, linestyle=(0,(20,10)))
                ax3.plot(x_g, y_g, z_g, 'mediumpurple', lw=3, alpha=0.8, label='pred')
                ax3.plot(x_g, y_g, z_g, 'white', lw=1, alpha=0.8, linestyle=(0,(20,10)))
            elif lane_cate == 2: # white solid
                ax1.plot(x_2d, y_2d, 'mediumturquoise', lw=3, alpha=0.8, label='pred')
                ax1.plot(x_2d, y_2d, 'white', lw=1, alpha=0.5)
                ax2.plot(x_ipm, y_ipm, 'mediumturquoise', lw=3, alpha=0.8, label='pred')
                ax2.plot(x_ipm, y_ipm, 'white', lw=1, alpha=0.5)
                ax3.plot(x_g, y_g, z_g, 'mediumturquoise', lw=3, alpha=0.8, label='pred')
                ax3.plot(x_g, y_g, z_g, 'white', lw=1, alpha=0.5)
            elif lane_cate == 3: # double-white-dash
                ax1.plot(x_2d, y_2d, 'mediumorchid', lw=4, alpha=0.8, label='pred')
                ax1.plot(np.array(x_2d)+0.15, y_2d, 'white', lw=0.5, alpha=1, linestyle=(0,(20,10)))
                ax1.plot(np.array(x_2d)-0.15, y_2d, 'white', lw=0.5, alpha=1, linestyle=(0,(20,10)))
                ax2.plot(x_ipm, y_ipm, 'mediumorchid', lw=4, alpha=0.8, label='pred')
                ax2.plot(np.array(x_ipm)+0.15, y_ipm, 'white', lw=0.5, alpha=1, linestyle=(0,(20,10)))
                ax2.plot(np.array(x_ipm)-0.15, y_ipm, 'white', lw=0.5, alpha=1, linestyle=(0,(20,10)))
                ax3.plot(x_g, y_g, z_g, 'mediumorchid', lw=4, alpha=0.8, label='pred')
                ax3.plot(np.array(x_g)+0.15, y_g, z_g, 'white', lw=0.5, alpha=1, linestyle=(0,(20,10)))
                ax3.plot(np.array(x_g)-0.15, y_g, z_g, 'white', lw=0.5, alpha=1, linestyle=(0,(20,10)))
            elif lane_cate == 4: # double-white-solid
                ax1.plot(x_2d, y_2d, 'lightskyblue', lw=4, alpha=0.8, label='pred')
                ax1.plot(np.array(x_2d)+0.15, y_2d, 'white', lw=0.5, alpha=1)
                ax1.plot(np.array(x_2d)-0.15, y_2d, 'white', lw=0.5, alpha=1)
                ax2.plot(x_ipm, y_ipm, 'lightskyblue', lw=4, alpha=0.8, label='pred')
                ax2.plot(np.array(x_ipm)+0.15, y_ipm, 'white', lw=0.5, alpha=1)
                ax2.plot(np.array(x_ipm)-0.15, y_ipm, 'white', lw=0.5, alpha=1)
                ax3.plot(x_g, y_g, z_g, 'lightskyblue', lw=4, alpha=0.8, label='pred')
                ax3.plot(np.array(x_g)+0.15, y_g, z_g, 'white', lw=0.5, alpha=1)
                ax3.plot(np.array(x_g)-0.15, y_g, z_g, 'white', lw=0.5, alpha=1)
            elif lane_cate == 5: # white-ldash-rsolid
                ax1.plot(x_2d, y_2d, 'hotpink', lw=4, alpha=0.8, label='pred')
                ax1.plot(np.array(x_2d)+0.15, y_2d, 'white', lw=0.5, alpha=1, linestyle=(0,(20,10)))
                ax1.plot(np.array(x_2d)-0.15, y_2d, 'white', lw=0.5, alpha=1)
                ax2.plot(x_ipm, y_ipm, 'hotpink', lw=4, alpha=0.8, label='pred')
                ax2.plot(np.array(x_ipm)+0.15, y_ipm, 'white', lw=0.5, alpha=1, linestyle=(0,(20,10)))
                ax2.plot(np.array(x_ipm)-0.15, y_ipm, 'white', lw=0.5, alpha=1)
                ax3.plot(x_g, y_g, z_g, 'hotpink', lw=4, alpha=0.8, label='pred')
                ax3.plot(np.array(x_g)+0.15, y_g, z_g, 'white', lw=0.5, alpha=1, linestyle=(0,(20,10)))
                ax3.plot(np.array(x_g)-0.15, y_g, z_g, 'white', lw=0.5, alpha=1)
            elif lane_cate == 6: # white-lsolid-rdash
                ax1.plot(x_2d, y_2d, 'cornflowerblue', lw=4, alpha=0.8, label='pred')
                ax1.plot(np.array(x_2d)+0.15, y_2d, 'white', lw=0.5, alpha=1)
                ax1.plot(np.array(x_2d)-0.15, y_2d, 'white', lw=0.5, alpha=1, linestyle=(0,(20,10)))
                ax2.plot(x_ipm, y_ipm, 'cornflowerblue', lw=4, alpha=0.8, label='pred')
                ax2.plot(np.array(x_ipm)+0.15, y_ipm, 'white', lw=0.5, alpha=1)
                ax2.plot(np.array(x_ipm)-0.15, y_ipm, 'white', lw=0.5, alpha=1, linestyle=(0,(20,10)))
                ax3.plot(x_g, y_g, z_g, 'cornflowerblue', lw=4, alpha=0.8, label='pred')
                ax3.plot(np.array(x_g)+0.15, y_g, z_g, 'white', lw=0.5, alpha=1)
                ax3.plot(np.array(x_g)-0.15, y_g, z_g, 'white', lw=0.5, alpha=1, linestyle=(0,(20,10)))
            elif lane_cate == 7: # yellow-dash
                ax1.plot(x_2d, y_2d, 'lawngreen', lw=3, alpha=0.8)
                ax1.plot(x_2d, y_2d, 'white', lw=1, alpha=0.8, linestyle=(0,(20,10)))
                ax2.plot(x_ipm, y_ipm, 'lawngreen', lw=3, alpha=0.8)
                ax2.plot(x_ipm, y_ipm, 'white', lw=1, alpha=0.8, linestyle=(0,(20,10)))
                ax3.plot(x_g, y_g, z_g, 'lawngreen', lw=3, alpha=0.8)
                ax3.plot(x_g, y_g, z_g, 'white', lw=1, alpha=0.8, linestyle=(0,(20,10)))
            elif lane_cate == 8: # yellow-solid
                ax1.plot(x_2d, y_2d, 'dodgerblue', lw=3, alpha=0.8)
                ax1.plot(x_2d, y_2d, 'white', lw=1, alpha=0.5)
                ax2.plot(x_ipm, y_ipm, 'dodgerblue', lw=3, alpha=0.8)
                ax2.plot(x_ipm, y_ipm, 'white', lw=1, alpha=0.5)
                ax3.plot(x_g, y_g, z_g, 'dodgerblue', lw=3, alpha=0.8)
                ax3.plot(x_g, y_g, z_g, 'white', lw=1, alpha=0.5)
            elif lane_cate == 9: # double-yellow-dash
                ax1.plot(x_2d, y_2d, 'salmon', lw=4, alpha=0.8, label='pred')
                ax1.plot(np.array(x_2d)+0.15, y_2d, 'white', lw=0.5, alpha=1, linestyle=(0,(20,10)))
                ax1.plot(np.array(x_2d)-0.15, y_2d, 'white', lw=0.5, alpha=1, linestyle=(0,(20,10)))
                ax2.plot(x_ipm, y_ipm, 'salmon', lw=4, alpha=0.8, label='pred')
                ax2.plot(np.array(x_ipm)+0.15, y_ipm, 'white', lw=0.5, alpha=1, linestyle=(0,(20,10)))
                ax2.plot(np.array(x_ipm)-0.15, y_ipm, 'white', lw=0.5, alpha=1, linestyle=(0,(20,10)))
                ax3.plot(x_g, y_g, z_g, 'salmon', lw=4, alpha=0.8, label='pred')
                ax3.plot(np.array(x_g)+0.15, y_g, z_g, 'white', lw=0.5, alpha=1, linestyle=(0,(20,10)))
                ax3.plot(np.array(x_g)-0.15, y_g, z_g, 'white', lw=0.5, alpha=1, linestyle=(0,(20,10)))
            elif lane_cate == 10: # double-yellow-solid
                ax1.plot(x_2d, y_2d, 'lightcoral', lw=4, alpha=0.8, label='pred')
                ax1.plot(np.array(x_2d)+0.15, y_2d, 'white', lw=0.5, alpha=1)
                ax1.plot(np.array(x_2d)-0.15, y_2d, 'white', lw=0.5, alpha=1)
                ax2.plot(x_ipm, y_ipm, 'lightcoral', lw=4, alpha=0.8, label='pred')
                ax2.plot(np.array(x_ipm)+0.15, y_ipm, 'white', lw=0.5, alpha=1)
                ax2.plot(np.array(x_ipm)-0.15, y_ipm, 'white', lw=0.5, alpha=1)
                ax3.plot(x_g, y_g, z_g, 'lightcoral', lw=4, alpha=0.8, label='pred')
                ax3.plot(np.array(x_g)+0.15, y_g, z_g, 'white', lw=0.5, alpha=1)
                ax3.plot(np.array(x_g)-0.15, y_g, z_g, 'white', lw=0.5, alpha=1)
            elif lane_cate == 11: # yellow-ldash-rsolid
                ax1.plot(x_2d, y_2d, 'coral', lw=4, alpha=0.8, label='pred')
                ax1.plot(np.array(x_2d)+0.15, y_2d, 'white', lw=0.5, alpha=1, linestyle=(0,(20,10)))
                ax1.plot(np.array(x_2d)-0.15, y_2d, 'white', lw=0.5, alpha=1)
                ax2.plot(x_ipm, y_ipm, 'coral', lw=4, alpha=0.8, label='pred')
                ax2.plot(np.array(x_ipm)+0.15, y_ipm, 'white', lw=0.5, alpha=1, linestyle=(0,(20,10)))
                ax2.plot(np.array(x_ipm)-0.15, y_ipm, 'white', lw=0.5, alpha=1)
                ax3.plot(x_g, y_g, z_g, 'coral', lw=4, alpha=0.8, label='pred')
                ax3.plot(np.array(x_g)+0.15, y_g, z_g, 'white', lw=0.5, alpha=1, linestyle=(0,(20,10)))
                ax3.plot(np.array(x_g)-0.15, y_g, z_g, 'white', lw=0.5, alpha=1)
            elif lane_cate == 12: # yellow-lsolid-rdash
                ax1.plot(x_2d, y_2d, 'lightseagreen', lw=4, alpha=0.8, label='pred')
                ax1.plot(np.array(x_2d)+0.15, y_2d, 'white', lw=0.5, alpha=1)
                ax1.plot(np.array(x_2d)-0.15, y_2d, 'white', lw=0.5, alpha=1, linestyle=(0,(20,10)))
                ax2.plot(x_ipm, y_ipm, 'lightseagreen', lw=4, alpha=0.8, label='pred')
                ax2.plot(np.array(x_ipm)+0.15, y_ipm, 'white', lw=0.5, alpha=1)
                ax2.plot(np.array(x_ipm)-0.15, y_ipm, 'white', lw=0.5, alpha=1, linestyle=(0,(20,10)))
                ax3.plot(x_g, y_g, z_g, 'lightseagreen', lw=4, alpha=0.8, label='pred')
                ax3.plot(np.array(x_g)+0.15, y_g, z_g, 'white', lw=0.5, alpha=1)
                ax3.plot(np.array(x_g)-0.15, y_g, z_g, 'white', lw=0.5, alpha=1, linestyle=(0,(20,10)))
            elif lane_cate == 13: # fishbone
                ax1.plot(x_2d, y_2d, 'royalblue', lw=3, alpha=0.8)
                ax1.plot(x_2d, y_2d, 'white', lw=1, alpha=0.5)
                ax2.plot(x_ipm, y_ipm, 'royalblue', lw=3, alpha=0.8)
                ax2.plot(x_ipm, y_ipm, 'white', lw=1, alpha=0.5)
                ax3.plot(x_g, y_g, z_g, 'royalblue', lw=3, alpha=0.8)
                ax3.plot(x_g, y_g, z_g, 'white', lw=1, alpha=0.5)
            elif lane_cate == 14: # others
                ax1.plot(x_2d, y_2d, 'forestgreen', lw=3, alpha=0.8)
                ax1.plot(x_2d, y_2d, 'white', lw=1, alpha=0.5)
                ax2.plot(x_ipm, y_ipm, 'forestgreen', lw=3, alpha=0.8)
                ax2.plot(x_ipm, y_ipm, 'white', lw=1, alpha=0.5)
                ax3.plot(x_g, y_g, z_g, 'forestgreen', lw=3, alpha=0.8)
                ax3.plot(x_g, y_g, z_g, 'white', lw=1, alpha=0.5)

            elif lane_cate == 20 or lane_cate == 21: # road
                ax1.plot(x_2d, y_2d, 'gold', lw=2, alpha=0.8)
                ax1.plot(x_2d, y_2d, 'white', lw=1, alpha=0.5)
                ax2.plot(x_ipm, y_ipm, 'gold', lw=2, alpha=0.8)
                ax2.plot(x_ipm, y_ipm, 'white', lw=1, alpha=0.5)
                ax3.plot(x_g, y_g, z_g, 'gold', lw=2, alpha=0.8)
                ax3.plot(x_g, y_g, z_g, 'white', lw=1, alpha=0.5)
            else:
                ax1.plot(x_2d, y_2d, lw=3, alpha=0.8)
                ax1.plot(x_2d, y_2d, 'white', lw=1, alpha=0.5)
                ax2.plot(x_ipm, y_ipm, lw=3, alpha=0.8)
                ax2.plot(x_ipm, y_ipm, 'white', lw=1, alpha=0.5)
                ax3.plot(x_g, y_g, z_g, lw=3, alpha=0.8)
                ax3.plot(x_g, y_g, z_g, 'white', lw=1, alpha=0.5)

        for i in range(cnt_gt):
            x_values = gt_lanes[i][:, 0]
            z_values = gt_lanes[i][:, 1]
            lane_cate = gt_category[i]
            x_g = x_values[np.where(gt_visibility_mat[i, :])]
            if len(x_g) == 0:
                continue
            y_g = self.y_samples[np.where(gt_visibility_mat[i, :])]
            z_g = z_values[np.where(gt_visibility_mat[i, :])]
            fit1 = np.polyfit(y_g, x_g, 2)
            fit2 = np.polyfit(y_g, z_g,2)
            f_xy = np.poly1d(fit1)
            f_zy = np.poly1d(fit2)
            y_g = np.linspace(min(y_g), max(y_g), 5 * len(y_g))
            x_g = f_xy(y_g)
            z_g = f_zy(y_g)
            ax3.plot(x_g, y_g, z_g, lw=2, c='red', alpha=1, label='gt')

            x_2d, y_2d = projective_transformation(P_gt, x_g, y_g, z_g)
            valid_mask_2d = np.logical_and(np.logical_and(x_2d >= 0, x_2d < self.resize_w), np.logical_and(y_2d >= 0, y_2d < self.resize_h))
            x_2d = x_2d[valid_mask_2d]
            y_2d = y_2d[valid_mask_2d]
            ax1.plot(x_2d, y_2d, lw=2, c='red', alpha=1, label='gt')

            x_ipm, y_ipm = homographic_transformation(H_g2ipm, x_g, y_g)
            valid_mask_ipm = np.logical_and(np.logical_and(x_ipm >= 0, x_ipm < self.ipm_w), np.logical_and(y_ipm >= 0, y_ipm < self.ipm_h))
            x_ipm = x_ipm[valid_mask_ipm]
            y_ipm = y_ipm[valid_mask_ipm]
            ax2.plot(x_ipm, y_ipm, lw=2, c='red', alpha=1, label='gt')

        bottom, top = ax3.get_zlim()
        left, right = ax3.get_xlim()
        ax3.set_zlim(min(bottom, -0.1), max(top, 0.1))
        ax3.set_xlim(left, right)
        ax3.set_ylim(vis_min_y, vis_max_y)
        ax3.locator_params(nbins=5, axis='x')
        ax3.locator_params(nbins=10, axis='z')
        ax3.tick_params(pad=18, labelsize=15)
        print("save to", ops.join(save_dir, img_name.replace("/", "_")))
        fig.savefig(ops.join(save_dir, img_name.replace("/", "_")))
        plt.close(fig)

    #这个vis能不能改为train的
    def visualize(self, pred_file, gt_file, test_file=None, prob_th=0.5, img_dir=None, save_dir=None, vis_step=20):
        mmcv.mkdir_or_exist(save_dir)
        if pred_file is not None:#同时绘制gt和pred
            pred_lines = open(pred_file).readlines()
            json_pred = [json.loads(line) for line in pred_lines]
            json_gt = [json.loads(line) for line in open(gt_file).readlines()]
            if test_file is not None:
                test_list = [s.strip().split('.')[0] for s in open(test_file, 'r').readlines()]
                json_pred = [s  for s in json_pred if s['file_path'][:-4] in test_list]
                json_gt = [s for s in json_gt if s['file_path'][:-4] in test_list]
            if len(json_gt) != len(json_pred):
                warnings.warn('We do not get the predictions of all the test tasks')
            gts = {l['file_path']: l for l in json_gt}
            for i, pred in tqdm(enumerate(json_pred)):
                if i % vis_step > 0:
                    continue
                raw_file = pred['file_path']
                gt = gts[raw_file]
                self.vis(gt, pred, save_dir, img_dir, raw_file, prob_th)
        else:#仅绘制gt,后面改成多线程得了
            json_gt = [json.loads(line) for line in open(gt_file).readlines()]
            # json_gt = self.load_json_lines_parallel(gt_file, workers=worker)# 作用不大，速度依然很慢
            end = time.time()
            print('load json gt time:', end-start,'with workers:', worker)
            print('json_gt:', len(json_gt))
            if test_file is not None:
                test_list = [s.strip().split('.')[0] for s in open(test_file, 'r').readlines()]
                json_gt = [s for s in json_gt if s['file_path'][:-4] in test_list]
            gts = {l['file_path']: l for l in json_gt}
            depth =dict()
            try:
                depth['device']= torWch.device("cuda" if torch.cuda.is_available() else "cpu")
                depth['model'] = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(depth['device']).eval() #网不好就用不了,特别离谱
                depth['transforms'] = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
                # 这东西读取不到很正常,网络老是有问题
            except:
                print("depth model load failed, try next time!")
                depth = None


            for i, gt in tqdm(enumerate(json_gt)):#用一下多线程
                if gt['file_path'] != 'training/segment-11392401368700458296_1086_429_1106_429_with_camera_labels/150913982702468600.jpg':
                    continue #只要这个快速遍历一下 #109419
                else:
                # if i % vis_step > 0: #通过上面这个遍历
                #     continue
                    print(f'find target in json_gt[{i}]!')
                    raw_file = gt['file_path']
                    self.vis_raw(gt, save_dir, img_dir, raw_file, prob_th,depth) #有问题
    def load_json_lines_parallel(self,gt_file, workers=None):
        import json
        from multiprocessing import Pool, cpu_count
        import mmap
        """
        多进程解析JSON Lines文件（每行一个独立对象）
        
        参数：
            gt_file (str): JSON Lines文件路径
            workers (int): 进程数（默认使用CPU核心数）
        """
        # 使用内存映射加速大文件读取
        with open(gt_file, "r+") as f:
            # 创建内存映射文件
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            # 按行分割（比 readlines() 更高效）
            lines = mm.read().splitlines()
            mm.close()
        
        # 过滤空行
        lines = [line for line in lines if line.strip()]
        
        # 启动多进程池
        workers = workers or cpu_count()
        with Pool(processes=workers) as pool:
            results = pool.map(parse_line, lines, chunksize=1000)
        
        # 过滤解析失败项
        return [res for res in results if res is not None]

    def vis_raw(self, gt, save_dir, img_dir, img_name, prob_th=0.5,depth=None):#其实就是没有pred的vis
        img_path = os.path.join(img_dir, 'images', img_name)
        fig = plt.figure()
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133, projection='3d')
        

        # 绘制补充文件
        fig2 = plt.figure()
        ax4 = fig2.add_subplot(121)
        ax5 = fig2.add_subplot(122)

        fig3 = plt.figure()
        ax6 = fig3.add_subplot(121)
        ax7 = fig3.add_subplot(122)
        # ax6 = fig2.add_subplot(133, projection='3d')
        img2 = np.zeros((360, 480, 3), dtype=np.uint8)
        # img2_ipm = img2_ipm #在上面进行了copy

        # 需要图形：1.纯IPM图，2.bev_seg图，3.
        # evaluate lanelines
        cam_extrinsics = np.array(gt['extrinsic'])
        # Re-calculate extrinsic matrix based on ground coordinate
        R_vg = np.array([[0, 1, 0],
                            [-1, 0, 0],
                            [0, 0, 1]], dtype=float)
        R_gc = np.array([[1, 0, 0],
                            [0, 0, 1],
                            [0, -1, 0]], dtype=float)
        cam_extrinsics[:3, :3] = np.matmul(np.matmul(
                                    np.matmul(np.linalg.inv(R_vg), cam_extrinsics[:3, :3]),
                                        R_vg), R_gc)

        cam_extrinsics[0:2, 3] = 0.0

        cam_intrinsics = gt['intrinsic']
        cam_intrinsics = np.array(cam_intrinsics)
        gt_lanes_packed = gt['lane_lines']

        gt_lanes, gt_visibility, gt_category = [], [], []
        for j, gt_lane_packed in enumerate(gt_lanes_packed):
            # A GT lane can be either 2D or 3D
            # if a GT lane is 3D, the height is intact from 3D GT, so keep it intact here too
            lane = np.array(gt_lane_packed['xyz'])
            lane_visibility = np.ones(len(lane[0]))

            lane = np.vstack((lane, np.ones((1, lane.shape[1]))))
            cam_representation = np.linalg.inv(
                                    np.array([[0, 0, 1, 0],
                                                [-1, 0, 0, 0],
                                                [0, -1, 0, 0],
                                                [0, 0, 0, 1]], dtype=float))

            lane = np.matmul(cam_extrinsics, np.matmul(cam_representation, lane))
            lane = lane[0:3, :].T

            gt_lanes.append(lane)
            gt_visibility.append(lane_visibility)
            gt_category.append(gt_lane_packed['category'])

        # only consider those gt lanes overlapping with sampling range
        gt_category = [gt_category[k] for k, lane in enumerate(gt_lanes)
                        if lane[0, 1] < self.y_samples[-1] and lane[-1, 1] > self.y_samples[0]]
        gt_lanes = [lane for lane in gt_lanes if lane[0, 1] < self.y_samples[-1] and lane[-1, 1] > self.y_samples[0]]

        gt_lanes = [prune_3d_lane_by_range(np.array(lane), self.x_min, self.x_max) for lane in gt_lanes]

        gt_category = [gt_category[k] for k, lane in enumerate(gt_lanes) if lane.shape[0] > 1]
        gt_lanes = [lane for lane in gt_lanes if lane.shape[0] > 1]

        cnt_gt = len(gt_lanes)

        P_g2im = projection_g2im_extrinsic(cam_extrinsics, cam_intrinsics)
        P_gt = np.matmul(self.H_crop, P_g2im)
        img = cv2.imread(img_path)
        img = cv2.warpPerspective(img, self.H_crop, (self.resize_w, self.resize_h))
        img = img.astype(np.float) / 255
        
        H_ipm2g = cv2.getPerspectiveTransform(
            np.float32([[0, 0], [self.ipm_w - 1, 0], [0, self.ipm_h - 1], [self.ipm_w - 1, self.ipm_h - 1]]), 
            np.float32(self.top_view_region))
        H_g2ipm = np.linalg.inv(H_ipm2g)
        H_g2im = homography_g2im_extrinsic(cam_extrinsics, cam_intrinsics)
        H_im2g = np.linalg.inv(H_g2im)
        H_im2ipm = np.linalg.inv(np.matmul(H_g2im, H_ipm2g))
        raw_img = cv2.imread(img_path)
        raw_img = raw_img.astype(np.float) / 255
        im_ipm = cv2.warpPerspective(raw_img, H_im2ipm, (self.ipm_w, self.ipm_h))
        im_ipm = np.clip(im_ipm, 0, 1)

        ax1.imshow(img[:, :, [2, 1, 0]])
        ax2.imshow(im_ipm[:, :, [2, 1, 0]])
        img2_ipm = im_ipm.copy()
        gt_visibility_mat = np.zeros((cnt_gt, 10))

            # # ATTENTION: ensure y mono increase before interpolation: but it can reduce size
            # pred_lanes[i] = make_lane_y_mono_inc(np.array(pred_lanes[i]))
            # pred_lane = prune_3d_lane_by_range(np.array(pred_lanes[i]), self.x_min, self.x_max)

        for i in range(cnt_gt):
            min_y = np.min(np.array(gt_lanes[i])[:, 1])
            max_y = np.max(np.array(gt_lanes[i])[:, 1])
            x_values, z_values, visibility_vec = resample_laneline_in_y(np.array(gt_lanes[i]), self.y_samples,
                                                                        out_vis=True)
            gt_lanes[i] = np.vstack([x_values, z_values]).T
            gt_visibility_mat[i, :] = np.logical_and(x_values >= self.x_min,
                                                     np.logical_and(x_values <= self.x_max,
                                                                    np.logical_and(self.y_samples >= min_y,
                                                                                   self.y_samples <= max_y)))
            gt_visibility_mat[i, :] = np.logical_and(gt_visibility_mat[i, :], visibility_vec)


        for i in range(cnt_gt):
            x_values = gt_lanes[i][:, 0]
            z_values = gt_lanes[i][:, 1]
            lane_cate = gt_category[i]
            x_g = x_values[np.where(gt_visibility_mat[i, :])]
            if len(x_g) == 0:
                continue
            y_g = self.y_samples[np.where(gt_visibility_mat[i, :])]
            z_g = z_values[np.where(gt_visibility_mat[i, :])]
            fit1 = np.polyfit(y_g, x_g, 2)
            fit2 = np.polyfit(y_g, z_g,2)
            f_xy = np.poly1d(fit1)
            f_zy = np.poly1d(fit2)
            y_g = np.linspace(min(y_g), max(y_g), 5 * len(y_g))
            x_g = f_xy(y_g)
            z_g = f_zy(y_g)
            ax3.plot(x_g, y_g, z_g, lw=2, c='red', alpha=1, label='gt')
            # 以y_g作为坐标
            x_2d, y_2d = projective_transformation(P_gt, x_g, y_g, z_g)
            valid_mask_2d = np.logical_and(np.logical_and(x_2d >= 0, x_2d < self.resize_w), np.logical_and(y_2d >= 0, y_2d < self.resize_h))
            x_2d = x_2d[valid_mask_2d]
            y_2d = y_2d[valid_mask_2d]
            ax1.plot(x_2d, y_2d, lw=2, c='red', alpha=1, label='gt')
            ax4.plot(x_2d, y_2d, lw=2, c='white', alpha=1, label='gt') #画出纯gt白线

            x_ipm, y_ipm = homographic_transformation(H_g2ipm, x_g, y_g)
            valid_mask_ipm = np.logical_and(np.logical_and(x_ipm >= 0, x_ipm < self.ipm_w), np.logical_and(y_ipm >= 0, y_ipm < self.ipm_h))
            x_ipm = x_ipm[valid_mask_ipm]
            y_ipm = y_ipm[valid_mask_ipm]
            ax2.plot(x_ipm, y_ipm, lw=2, c='red', alpha=1, label='gt')

        bottom, top = ax3.get_zlim()
        left, right = ax3.get_xlim()
        ax3.set_zlim(min(bottom, -0.1), max(top, 0.1))
        ax3.set_xlim(left, right)
        ax3.set_ylim(vis_min_y, vis_max_y)
        ax3.locator_params(nbins=5, axis='x')
        ax3.locator_params(nbins=10, axis='z')
        ax3.tick_params(pad=18, labelsize=15)
        print("save to", ops.join(save_dir, img_name.replace("/", "_")))
        path_part, dot, ext = img_name.rpartition('.')
        new_name = f"{path_part}.pdf"
        fig.savefig(ops.join(save_dir, new_name.replace("/", "_"))) #这个绘图失败了
        plt.close(fig)

        #再画一组图
        # 1.纯IPM图，2.bev_seg图，3.点的高度图 4.深度图
        #         

        ax4.imshow(img2[:, :, [2, 1, 0]])
        ax5.imshow(img2_ipm[:, :, [2, 1, 0]])

        path_part, dot, ext = img_name.rpartition('.')
        new_name = f"{path_part}_append1.pdf"
        print("save to", ops.join(save_dir, new_name.replace("/", "_")))
        fig2.savefig(ops.join(save_dir, new_name.replace("/", "_")))
        plt.close(fig)

        #再画一组图
        # 1.高度图 2.深度图
        # model_type = "MiDaS_Hybrid"  # 可选: DPT_Large, DPT_Hybrid, MiDaS_small,先用一个最小的
        model = depth['model']
        device = depth['device']
        input = cv2.imread(img_path)
        input = cv2.warpPerspective(input, self.H_crop, (self.resize_w, self.resize_h))
        # img = img.astype(np.float) / 255
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)

        # midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        # if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        #     transform = midas_transforms.dpt_transform
        # else:
        #     transform = midas_transforms.small_transform
        transform = depth['transforms']
        input_batch = transform(input).to(device)#目前测试通过的大小是1，3，192，256
        with torch.no_grad():
            prediction = model(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=input.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        output = prediction.detach().cpu().numpy()

        ax6.imshow(output)
        path_part, dot, ext = img_name.rpartition('.')
        new_name = f"{path_part}_depth.png" #把后缀改为pdf，否则用{ext}# 保持为
        maximum = np.max(output)
        minimum = np.min(output)
        # depth_map = output/maximum #用最大值做归一化,但是不需要
        depth_colored = cv2.applyColorMap(cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U),cv2.COLORMAP_MAGMA)
        # print("save to", ops.join(save_dir, new_name.replace("/", "_")))
        # fig3.savefig(ops.join(save_dir, new_name.replace("/", "_")))
        save_dir2= os.path.join(save_dir, "depth")
        cv2.imwrite(ops.join(save_dir2, new_name.replace("/", "_")), depth_colored)
        print('raw depth generated!')




        #深度图用fig3吧
        #最后按照名字单独存储每张图

#         depth_colored = cv2.applyColorMap(
#     cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U),
#     cv2.COLORMAP_MAGMA
# )

        # cv2.imwrite("depth_output.jpg", depth_colored)
        #接下来画登高图


        
        
        



        


