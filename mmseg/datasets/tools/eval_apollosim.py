"""
Description: This code is to evaluate 3D lane detection. The optimal matching between ground-truth set and predicted
set of lanes are sought via solving a min cost flow.

Evaluation metrics includes:
    Average Precision (AP)
    Max F-scores
    x error close (0 - 40 m)
    x error far (0 - 100 m)
    z error close (0 - 40 m)
    z error far (0 - 100 m)

Reference: "Gen-LaneNet: Generalized and Scalable Approach for 3D Lane Detection". Y. Guo. etal. 2020

Author: Yuliang Guo (33yuliangguo@gmail.com)
Date: March, 2020
"""

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

from .utils import *
from .MinCostFlow import SolveMinCostFlow
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (35, 30)
plt.rcParams.update({'font.size': 25})
plt.rcParams.update({'font.weight': 'semibold'})

color = [[0, 0, 255],  # red
         [0, 255, 0],  # green
         [255, 0, 255],  # purple
         [255, 255, 0]]  # cyan

vis_min_y = 5
vis_max_y = 80


class LaneEval(object):
    def __init__(self, db):
        self.dataset_dir = db.data_root
        self.K = db.K
        self.no_centerline = db.no_centerline
        self.resize_h = db.resize_h
        self.resize_w = db.resize_w
        # H_crop: [[rx, 0, 0], [0, ry, 0], [0, 0, 1]]
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
        self.y_samples = np.linspace(self.y_min, self.y_max, num=100, endpoint=False)
        self.dist_th = 1.5
        self.ratio_th = 0.75
        self.close_range = 40

    def bench(self, pred_lanes, gt_lanes, gt_visibility, raw_file, gt_cam_height, gt_cam_pitch):
        """
            Matching predicted lanes and ground-truth lanes in their IPM projection, ignoring z attributes.
            x error, y_error, and z error are all considered, although the matching does not rely on z
            The input of prediction and ground-truth lanes are in ground coordinate, x-right, y-forward, z-up
            The fundamental assumption is: 1. there are no two points from different lanes with identical x, y
                                              but different z's
                                           2. there are no two points from a single lane having identical x, y
                                              but different z's
            If the interest area is within the current drivable road, the above assumptions are almost always valid.

        :param pred_lanes: N X 2 or N X 3 lists depending on 2D or 3D
        :param gt_lanes: N X 2 or N X 3 lists depending on 2D or 3D
        :param raw_file: file path rooted in dataset folder
        :param gt_cam_height: camera height given in ground-truth data
        :param gt_cam_pitch: camera pitch given in ground-truth data
        :return:
        """

        # change this properly
        close_range_idx = np.where(self.y_samples > self.close_range)[0][0]

        r_lane, p_lane = 0., 0.
        x_error_close = []
        x_error_far = []
        z_error_close = []
        z_error_far = []

        # only keep the visible portion
        gt_lanes = [prune_3d_lane_by_visibility(np.array(gt_lane), np.array(gt_visibility[k])) for k, gt_lane in
                    enumerate(gt_lanes)]
        gt_lanes = [lane for lane in gt_lanes if lane.shape[0] > 1]
        # only consider those gt lanes overlapping with sampling range
        gt_lanes = [lane for lane in gt_lanes if lane[0, 1] < self.y_samples[-1] and lane[-1, 1] > self.y_samples[0]]
        gt_lanes = [prune_3d_lane_by_range(np.array(gt_lane), 3 * self.x_min, 3 * self.x_max) for gt_lane in gt_lanes]
        gt_lanes = [lane for lane in gt_lanes if lane.shape[0] > 1]
        cnt_gt = len(gt_lanes)
        cnt_pred = len(pred_lanes)

        gt_visibility_mat = np.zeros((cnt_gt, 100))
        pred_visibility_mat = np.zeros((cnt_pred, 100))
        # resample gt and pred at y_samples
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

        adj_mat = np.zeros((cnt_gt, cnt_pred), dtype=np.int)
        cost_mat = np.zeros((cnt_gt, cnt_pred), dtype=np.int)
        cost_mat.fill(1000)
        num_match_mat = np.zeros((cnt_gt, cnt_pred), dtype=np.float)
        x_dist_mat_close = np.zeros((cnt_gt, cnt_pred), dtype=np.float)
        x_dist_mat_close.fill(1000.)
        x_dist_mat_far = np.zeros((cnt_gt, cnt_pred), dtype=np.float)
        x_dist_mat_far.fill(1000.)
        z_dist_mat_close = np.zeros((cnt_gt, cnt_pred), dtype=np.float)
        z_dist_mat_close.fill(1000.)
        z_dist_mat_far = np.zeros((cnt_gt, cnt_pred), dtype=np.float)
        z_dist_mat_far.fill(1000.)
        # compute curve to curve distance
        for i in range(cnt_gt):
            for j in range(cnt_pred):
                x_dist = np.abs(gt_lanes[i][:, 0] - pred_lanes[j][:, 0])
                z_dist = np.abs(gt_lanes[i][:, 1] - pred_lanes[j][:, 1])
                euclidean_dist = np.sqrt(x_dist ** 2 + z_dist ** 2)

                # apply visibility to penalize different partial matching accordingly
                euclidean_dist[
                    np.logical_or(gt_visibility_mat[i, :] < 0.5, pred_visibility_mat[j, :] < 0.5)] = self.dist_th

                # if np.average(euclidean_dist) < 2*self.dist_th: # don't prune here to encourage finding perfect match
                num_match_mat[i, j] = np.sum(euclidean_dist < self.dist_th)
                adj_mat[i, j] = 1
                # ATTENTION: use the sum as int type to meet the requirements of min cost flow optimization (int type)
                # using num_match_mat as cost does not work?
                cost_mat[i, j] = np.sum(euclidean_dist).astype(np.int)
                # cost_mat[i, j] = num_match_mat[i, j]

                # use the both visible portion to calculate distance error
                both_visible_indices = np.logical_and(gt_visibility_mat[i, :] > 0.5, pred_visibility_mat[j, :] > 0.5)
                if np.sum(both_visible_indices[:close_range_idx]) > 0:
                    x_dist_mat_close[i, j] = np.sum(
                        x_dist[:close_range_idx] * both_visible_indices[:close_range_idx]) / np.sum(
                        both_visible_indices[:close_range_idx])
                    z_dist_mat_close[i, j] = np.sum(
                        z_dist[:close_range_idx] * both_visible_indices[:close_range_idx]) / np.sum(
                        both_visible_indices[:close_range_idx])
                else:
                    x_dist_mat_close[i, j] = self.dist_th
                    z_dist_mat_close[i, j] = self.dist_th

                if np.sum(both_visible_indices[close_range_idx:]) > 0:
                    x_dist_mat_far[i, j] = np.sum(
                        x_dist[close_range_idx:] * both_visible_indices[close_range_idx:]) / np.sum(
                        both_visible_indices[close_range_idx:])
                    z_dist_mat_far[i, j] = np.sum(
                        z_dist[close_range_idx:] * both_visible_indices[close_range_idx:]) / np.sum(
                        both_visible_indices[close_range_idx:])
                else:
                    x_dist_mat_far[i, j] = self.dist_th
                    z_dist_mat_far[i, j] = self.dist_th

        # solve bipartite matching vis min cost flow solver
        match_results = SolveMinCostFlow(adj_mat, cost_mat)
        match_results = np.array(match_results)

        # only a match with avg cost < self.dist_th is consider valid one
        match_gt_ids = []
        match_pred_ids = []
        if match_results.shape[0] > 0:
            for i in range(len(match_results)):
                if match_results[i, 2] < self.dist_th * self.y_samples.shape[0]:
                    gt_i = match_results[i, 0]
                    pred_i = match_results[i, 1]
                    # consider match when the matched points is above a ratio
                    if num_match_mat[gt_i, pred_i] / np.sum(gt_visibility_mat[gt_i, :]) >= self.ratio_th:
                        r_lane += 1
                        match_gt_ids.append(gt_i)
                    if num_match_mat[gt_i, pred_i] / np.sum(pred_visibility_mat[pred_i, :]) >= self.ratio_th:
                        p_lane += 1
                        match_pred_ids.append(pred_i)
                    x_error_close.append(x_dist_mat_close[gt_i, pred_i])
                    x_error_far.append(x_dist_mat_far[gt_i, pred_i])
                    z_error_close.append(z_dist_mat_close[gt_i, pred_i])
                    z_error_far.append(z_dist_mat_far[gt_i, pred_i])

        return r_lane, p_lane, cnt_gt, cnt_pred, x_error_close, x_error_far, z_error_close, z_error_far

    # compare predicted set and ground-truth set using a fixed lane probability threshold
    def bench_one_submit(self, pred_file, gt_file, prob_th=0.5):
        pred_lines = open(pred_file).readlines()
        json_pred = [json.loads(line) for line in pred_lines]
        json_gt = [json.loads(line) for line in open(gt_file).readlines()]
        
        # 将长度处理为一致
        json_gt2  = []
        for pred in json_pred:
            for gt in json_gt:
                if pred['raw_file'] == gt['raw_file']:
                    json_gt2.append(gt)
        print(f'pred_file: {pred_file}')
        print(f'gt_file: {gt_file}')
        json_gt = json_gt2
        
        if len(json_gt) != len(json_pred):
            raise Exception('We do not get the predictions of all the test tasks')
        gts = {l['raw_file']: l for l in json_gt}

        laneline_stats = []
        laneline_x_error_close = []
        laneline_x_error_far = []
        laneline_z_error_close = []
        laneline_z_error_far = []
        centerline_stats = []
        centerline_x_error_close = []
        centerline_x_error_far = []
        centerline_z_error_close = []
        centerline_z_error_far = []
        x_error_close_dict = {}
        z_error_close_dict = {}
        x_error_far_dict = {}
        z_error_far_dict = {}
        for i, pred in enumerate(json_pred):
            if 'raw_file' not in pred or 'laneLines' not in pred:
                raise Exception('raw_file or lanelines not in some predictions.')
            raw_file = pred['raw_file']

            pred_lanelines = pred['laneLines']
            pred_laneLines_prob = pred['laneLines_prob']
            pred_lanelines = [pred_lanelines[ii] for ii in range(len(pred_laneLines_prob)) if
                              pred_laneLines_prob[ii] > prob_th]

            if raw_file not in gts:
                raise Exception('Some raw_file from your predictions do not exist in the test tasks.')
            gt = gts[raw_file]
            gt_cam_height = gt['cam_height']
            gt_cam_pitch = gt['cam_pitch']

            # evaluate lanelines
            gt_lanelines = gt['laneLines']
            gt_visibility = gt['laneLines_visibility']
            # N to N matching of lanelines
            r_lane, p_lane, cnt_gt, cnt_pred, \
            x_error_close, x_error_far, \
            z_error_close, z_error_far = self.bench(pred_lanelines,
                                                    gt_lanelines,
                                                    gt_visibility,
                                                    raw_file,
                                                    gt_cam_height,
                                                    gt_cam_pitch)
            laneline_stats.append(np.array([r_lane, p_lane, cnt_gt, cnt_pred]))
            # consider x_error z_error only for the matched lanes
            laneline_x_error_close.extend(x_error_close)
            laneline_x_error_far.extend(x_error_far)
            laneline_z_error_close.extend(z_error_close)
            laneline_z_error_far.extend(z_error_far)
            x_error_close_dict[raw_file] = x_error_close
            z_error_close_dict[raw_file] = z_error_close
            x_error_far_dict[raw_file] = x_error_far
            z_error_far_dict[raw_file] = z_error_far

            # evaluate centerlines
            if not self.no_centerline:
                pred_centerlines = pred['centerLines']
                pred_centerlines_prob = pred['centerLines_prob']
                pred_centerlines = [pred_centerlines[ii] for ii in range(len(pred_centerlines_prob)) if
                                    pred_centerlines_prob[ii] > prob_th]

                gt_centerlines = gt['centerLines']
                gt_visibility = gt['centerLines_visibility']

                # N to N matching of lanelines
                r_lane, p_lane, cnt_gt, cnt_pred, \
                x_error_close, x_error_far, \
                z_error_close, z_error_far = self.bench(pred_centerlines,
                                                        gt_centerlines,
                                                        gt_visibility,
                                                        raw_file,
                                                        gt_cam_height,
                                                        gt_cam_pitch)
                centerline_stats.append(np.array([r_lane, p_lane, cnt_gt, cnt_pred]))
                # consider x_error z_error only for the matched lanes
                # if r_lane > 0 and p_lane > 0:
                centerline_x_error_close.extend(x_error_close)
                centerline_x_error_far.extend(x_error_far)
                centerline_z_error_close.extend(z_error_close)
                centerline_z_error_far.extend(z_error_far)

        output_stats = []
        laneline_stats = np.array(laneline_stats)
        laneline_x_error_close = np.array(laneline_x_error_close)
        laneline_x_error_far = np.array(laneline_x_error_far)
        laneline_z_error_close = np.array(laneline_z_error_close)
        laneline_z_error_far = np.array(laneline_z_error_far)

        R_lane = np.sum(laneline_stats[:, 0]) / (np.sum(laneline_stats[:, 2]) + 1e-6)
        P_lane = np.sum(laneline_stats[:, 1]) / (np.sum(laneline_stats[:, 3]) + 1e-6)
        F_lane = 2 * R_lane * P_lane / (R_lane + P_lane + 1e-6)
        x_error_close_avg = np.average(laneline_x_error_close)
        x_error_far_avg = np.average(laneline_x_error_far)
        z_error_close_avg = np.average(laneline_z_error_close)
        z_error_far_avg = np.average(laneline_z_error_far)

        output_stats.append(F_lane)
        output_stats.append(R_lane)
        output_stats.append(P_lane)
        output_stats.append(x_error_close_avg)
        output_stats.append(x_error_far_avg)
        output_stats.append(z_error_close_avg)
        output_stats.append(z_error_far_avg)
        output_stats.append(x_error_close_dict)
        output_stats.append(x_error_far_dict)
        output_stats.append(z_error_close_dict)
        output_stats.append(z_error_far_dict)

        if not self.no_centerline:
            centerline_stats = np.array(centerline_stats)
            centerline_x_error_close = np.array(centerline_x_error_close)
            centerline_x_error_far = np.array(centerline_x_error_far)
            centerline_z_error_close = np.array(centerline_z_error_close)
            centerline_z_error_far = np.array(centerline_z_error_far)

            R_lane = np.sum(centerline_stats[:, 0]) / (np.sum(centerline_stats[:, 2]) + 1e-6)
            P_lane = np.sum(centerline_stats[:, 1]) / (np.sum(centerline_stats[:, 3]) + 1e-6)
            F_lane = 2 * R_lane * P_lane / (R_lane + P_lane + 1e-6)
            x_error_close_avg = np.average(centerline_x_error_close)
            x_error_far_avg = np.average(centerline_x_error_far)
            z_error_close_avg = np.average(centerline_z_error_close)
            z_error_far_avg = np.average(centerline_z_error_far)

            output_stats.append(F_lane)
            output_stats.append(R_lane)
            output_stats.append(P_lane)
            output_stats.append(x_error_close_avg)
            output_stats.append(x_error_far_avg)
            output_stats.append(z_error_close_avg)
            output_stats.append(z_error_far_avg)

        return output_stats

    def bench_PR(self, pred_lanes, gt_lanes, gt_visibility):
        """
            Matching predicted lanes and ground-truth lanes in their IPM projection, ignoring z attributes.
            x error, y_error, and z error are all considered, although the matching does not rely on z
            The input of prediction and ground-truth lanes are in ground coordinate, x-right, y-forward, z-up
            The fundamental assumption is: 1. there are no two points from different lanes with identical x, y
                                              but different z's
                                           2. there are no two points from a single lane having identical x, y
                                              but different z's
            If the interest area is within the current drivable road, the above assumptions are almost always valid.

        :param pred_lanes: N X 2 or N X 3 lists depending on 2D or 3D
        :param gt_lanes: N X 2 or N X 3 lists depending on 2D or 3D
        :return:
        """

        r_lane, p_lane = 0., 0.

        # only keep the visible portion
        gt_lanes = [prune_3d_lane_by_visibility(np.array(gt_lane), np.array(gt_visibility[k])) for k, gt_lane in
                    enumerate(gt_lanes)]
        gt_lanes = [lane for lane in gt_lanes if lane.shape[0] > 1]
        # only consider those gt lanes overlapping with sampling range
        gt_lanes = [lane for lane in gt_lanes if lane[0, 1] < self.y_samples[-1] and lane[-1, 1] > self.y_samples[0]]
        gt_lanes = [prune_3d_lane_by_range(np.array(gt_lane), 3 * self.x_min, 3 * self.x_max) for gt_lane in gt_lanes]
        gt_lanes = [lane for lane in gt_lanes if lane.shape[0] > 1]
        cnt_gt = len(gt_lanes)
        cnt_pred = len(pred_lanes)

        gt_visibility_mat = np.zeros((cnt_gt, 100))
        pred_visibility_mat = np.zeros((cnt_pred, 100))
        # resample gt and pred at y_samples
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
            # pred_visibility_mat[i, :] = np.logical_and(x_values >= self.x_min, x_values <= self.x_max)

        adj_mat = np.zeros((cnt_gt, cnt_pred), dtype=np.int)
        cost_mat = np.zeros((cnt_gt, cnt_pred), dtype=np.int)
        cost_mat.fill(1000)
        num_match_mat = np.zeros((cnt_gt, cnt_pred), dtype=np.float)
        # compute curve to curve distance
        for i in range(cnt_gt):
            for j in range(cnt_pred):
                x_dist = np.abs(gt_lanes[i][:, 0] - pred_lanes[j][:, 0])
                z_dist = np.abs(gt_lanes[i][:, 1] - pred_lanes[j][:, 1])
                euclidean_dist = np.sqrt(x_dist ** 2 + z_dist ** 2)

                # apply visibility to penalize different partial matching accordingly
                euclidean_dist[
                    np.logical_or(gt_visibility_mat[i, :] < 0.5, pred_visibility_mat[j, :] < 0.5)] = self.dist_th

                # if np.average(euclidean_dist) < 2*self.dist_th: # don't prune here to encourage finding perfect match
                num_match_mat[i, j] = np.sum(euclidean_dist < self.dist_th)
                adj_mat[i, j] = 1
                # ATTENTION: use the sum as int type to meet the requirements of min cost flow optimization (int type)
                # why using num_match_mat as cost does not work?
                cost_mat[i, j] = np.sum(euclidean_dist).astype(np.int)
                # cost_mat[i, j] = num_match_mat[i, j]

        # solve bipartite matching vis min cost flow solver
        match_results = SolveMinCostFlow(adj_mat, cost_mat)
        match_results = np.array(match_results)

        # only a match with avg cost < self.dist_th is consider valid one
        match_gt_ids = []
        match_pred_ids = []
        if match_results.shape[0] > 0:
            for i in range(len(match_results)):
                if match_results[i, 2] < self.dist_th * self.y_samples.shape[0]:
                    gt_i = match_results[i, 0]
                    pred_i = match_results[i, 1]
                    # consider match when the matched points is above a ratio
                    if num_match_mat[gt_i, pred_i] / np.sum(gt_visibility_mat[gt_i, :]) >= self.ratio_th:
                        r_lane += 1
                        match_gt_ids.append(gt_i)
                    if num_match_mat[gt_i, pred_i] / np.sum(pred_visibility_mat[pred_i, :]) >= self.ratio_th:
                        p_lane += 1
                        match_pred_ids.append(pred_i)

        return r_lane, p_lane, cnt_gt, cnt_pred

    # evaluate two dataset at varying lane probability threshold to calculate AP
    def bench_one_submit_varying_probs(self, pred_file, gt_file):
        varying_th = np.linspace(0.05, 0.95, 19)   #
        # varying_th = np.linspace(0.25, 0.3, 2)
        # varying_th = np.linspace(0.75, 0.95, 5)   #
        # try:
        pred_lines = open(pred_file).readlines()
        json_pred = [json.loads(line) for line in pred_lines]
        # except BaseException as e:
        #     raise Exception('Fail to load json file of the prediction.')
        
        json_gt = [json.loads(line) for line in open(gt_file).readlines()]
        
        json_gt2  = []
        for pred in json_pred:
            for gt in json_gt:
                if pred['raw_file'] == gt['raw_file']:
                    json_gt2.append(gt)
        print(f'pred_file: {pred_file}')
        print(f'gt_file: {gt_file}')
        json_gt = json_gt2
        
        if len(json_gt) != len(json_pred):
            # raise Exception('We do not get the predictions of all the test tasks')
            print('We do not get the predictions of all the test tasks')
        gts = {l['raw_file']: l for l in json_gt}

        laneline_r_all = []
        laneline_p_all = []
        laneline_gt_cnt_all = []
        laneline_pred_cnt_all = []
        centerline_r_all = []
        centerline_p_all = []
        centerline_gt_cnt_all = []
        centerline_pred_cnt_all = []
        for i in tqdm(range(0, len(json_pred)), ncols=60, desc="Evaluating sample"):
            pred = json_pred[i]
        # for i, pred in enumerate(json_pred):
        #     print('Evaluating sample {} / {}'.format(i, len(json_pred)))
            if 'raw_file' not in pred or 'laneLines' not in pred:
                raise Exception('raw_file or lanelines not in some predictions.')
            raw_file = pred['raw_file']

            pred_lanelines = pred['laneLines']
            pred_laneLines_prob = pred['laneLines_prob']
            print(f'raw_file: {raw_file}')
            if raw_file not in gts:
                raise Exception('Some raw_file from your predictions do not exist in the test tasks.')
            gt = gts[raw_file]
            gt_cam_height = gt['cam_height']
            gt_cam_pitch = gt['cam_pitch']

            # evaluate lanelines
            gt_lanelines = gt['laneLines']
            gt_visibility = gt['laneLines_visibility']
            r_lane_vec = []
            p_lane_vec = []
            cnt_gt_vec = []
            cnt_pred_vec = []

            for prob_th in varying_th:
                pred_lanelines = [pred_lanelines[ii] for ii in range(len(pred_laneLines_prob)) if
                                  pred_laneLines_prob[ii] > prob_th]
                pred_laneLines_prob = [prob for prob in pred_laneLines_prob if prob > prob_th]
                pred_lanelines_copy = copy.deepcopy(pred_lanelines)
                # N to N matching of lanelines
                r_lane, p_lane, cnt_gt, cnt_pred = self.bench_PR(pred_lanelines_copy,
                                                                 gt_lanelines,
                                                                 gt_visibility)
                r_lane_vec.append(r_lane)
                p_lane_vec.append(p_lane)
                cnt_gt_vec.append(cnt_gt)
                cnt_pred_vec.append(cnt_pred)

            laneline_r_all.append(r_lane_vec)
            laneline_p_all.append(p_lane_vec)
            laneline_gt_cnt_all.append(cnt_gt_vec)
            laneline_pred_cnt_all.append(cnt_pred_vec)

            # evaluate centerlines
            if not self.no_centerline:
                pred_centerlines = pred['centerLines']
                pred_centerLines_prob = pred['centerLines_prob']
                gt_centerlines = gt['centerLines']
                gt_visibility = gt['centerLines_visibility']
                r_lane_vec = []
                p_lane_vec = []
                cnt_gt_vec = []
                cnt_pred_vec = []

                for prob_th in varying_th:
                    pred_centerlines = [pred_centerlines[ii] for ii in range(len(pred_centerLines_prob)) if
                                        pred_centerLines_prob[ii] > prob_th]
                    pred_centerLines_prob = [prob for prob in pred_centerLines_prob if prob > prob_th]
                    pred_centerlines_copy = copy.deepcopy(pred_centerlines)
                    # N to N matching of lanelines
                    r_lane, p_lane, cnt_gt, cnt_pred = self.bench_PR(pred_centerlines_copy,
                                                                     gt_centerlines,
                                                                     gt_visibility)
                    r_lane_vec.append(r_lane)
                    p_lane_vec.append(p_lane)
                    cnt_gt_vec.append(cnt_gt)
                    cnt_pred_vec.append(cnt_pred)
                centerline_r_all.append(r_lane_vec)
                centerline_p_all.append(p_lane_vec)
                centerline_gt_cnt_all.append(cnt_gt_vec)
                centerline_pred_cnt_all.append(cnt_pred_vec)

        output_stats = []
        # compute precision, recall
        laneline_r_all = np.array(laneline_r_all)
        laneline_p_all = np.array(laneline_p_all)
        laneline_gt_cnt_all = np.array(laneline_gt_cnt_all)
        laneline_pred_cnt_all = np.array(laneline_pred_cnt_all)

        R_lane = np.sum(laneline_r_all, axis=0) / (np.sum(laneline_gt_cnt_all, axis=0) + 1e-6)
        P_lane = np.sum(laneline_p_all, axis=0) / (np.sum(laneline_pred_cnt_all, axis=0) + 1e-6)
        F_lane = 2 * R_lane * P_lane / (R_lane + P_lane + 1e-6)

        output_stats.append(F_lane)
        output_stats.append(R_lane)
        output_stats.append(P_lane)

        if not self.no_centerline:
            centerline_r_all = np.array(centerline_r_all)
            centerline_p_all = np.array(centerline_p_all)
            centerline_gt_cnt_all = np.array(centerline_gt_cnt_all)
            centerline_pred_cnt_all = np.array(centerline_pred_cnt_all)

            R_lane = np.sum(centerline_r_all, axis=0) / (np.sum(centerline_gt_cnt_all, axis=0) + 1e-6)
            P_lane = np.sum(centerline_p_all, axis=0) / (np.sum(centerline_pred_cnt_all, axis=0) + 1e-6)
            F_lane = 2 * R_lane * P_lane / (R_lane + P_lane + 1e-6)

            output_stats.append(F_lane)
            output_stats.append(R_lane)
            output_stats.append(P_lane)
        else:
            output_stats.append(F_lane)
            output_stats.append(R_lane)
            output_stats.append(P_lane)

        # calculate metrics
        laneline_F = output_stats[0]
        laneline_F_max = np.max(laneline_F)
        laneline_max_i = np.argmax(laneline_F)
        laneline_R = output_stats[1]
        laneline_P = output_stats[2]
        centerline_F = output_stats[3]
        centerline_F_max = centerline_F[laneline_max_i]
        centerline_max_i = laneline_max_i
        centerline_R = output_stats[4]
        centerline_P = output_stats[5]

        laneline_R = np.array([1.] + laneline_R.tolist() + [0.])
        laneline_P = np.array([0.] + laneline_P.tolist() + [1.])
        centerline_R = np.array([1.] + centerline_R.tolist() + [0.])
        centerline_P = np.array([0.] + centerline_P.tolist() + [1.])
        f_laneline = interp1d(laneline_R, laneline_P)
        f_centerline = interp1d(centerline_R, centerline_P)
        r_range = np.linspace(0.05, 0.95, 19)
        laneline_AP = np.mean(f_laneline(r_range))
        centerline_AP = np.mean(f_centerline(r_range))

        json_out = {}
        json_out['laneline_R'] = laneline_R[1:-1].astype(np.float32).tolist()
        json_out['laneline_P'] = laneline_P[1:-1].astype(np.float32).tolist()
        json_out['laneline_F_max'] = laneline_F_max
        json_out['laneline_max_i'] = laneline_max_i.tolist()
        json_out['laneline_AP'] = laneline_AP

        json_out['centerline_R'] = centerline_R[1:-1].astype(np.float32).tolist()
        json_out['centerline_P'] = centerline_P[1:-1].astype(np.float32).tolist()
        json_out['centerline_F_max'] = centerline_F_max
        json_out['centerline_max_i'] = centerline_max_i.tolist()
        json_out['centerline_AP'] = centerline_AP

        json_out['max_F_prob_th'] = varying_th[laneline_max_i]

        return json_out


if __name__ == '__main__':
    vis = False
    parser = define_args()
    args = parser.parse_args()

    # two method are compared: '3D_LaneNet' and 'Gen_LaneNet'
    method_name = 'Gen_LaneNet_ext'

    # Three different splits of datasets: 'standard', 'rare_subsit', 'illus_chg'
    data_split = 'illus_chg'

    # location where the original dataset is saved. Image will be loaded in case of visualization
    args.dataset_dir = '~/Datasets/Apollo_Sim_3D_Lane_Release/'

    # load configuration for certain dataset
    sim3d_config(args)

    # auto-file in dependent paths
    gt_file = 'data_splits/' + data_split + '/test.json'
    pred_folder = 'data_splits/' + data_split + '/' + method_name
    pred_file = pred_folder + '/test_pred_file.json'

    # Initialize evaluator
    evaluator = LaneEval(args)

    # evaluation at varying thresholds
    eval_stats_pr = evaluator.bench_one_submit_varying_probs(pred_file, gt_file)
    max_f_prob = eval_stats_pr['max_F_prob_th']

    # evaluate at the point with max F-measure. Additional eval of position error. Option to visualize matching result
    eval_stats = evaluator.bench_one_submit(pred_file, gt_file, prob_th=max_f_prob, vis=vis)

    print("Metrics: AP, F-score, x error (close), x error (far), z error (close), z error (far)")
    print(
        "Laneline:  {:.3}  {:.3}  {:.3}  {:.3}  {:.3}  {:.3}".format(eval_stats_pr['laneline_AP'], eval_stats[0],
                                                                     eval_stats[3], eval_stats[4],
                                                                     eval_stats[5], eval_stats[6]))
    print("Centerline:  {:.3}  {:.3}  {:.3}  {:.3}  {:.3}  {:.3}".format(eval_stats_pr['centerline_AP'], eval_stats[7],
                                                                         eval_stats[10], eval_stats[11],
                                                                         eval_stats[12], eval_stats[13]))