import numpy as np
import cv2
from jarvis.eload import load_json
from munkres import Munkres
import os
from shapely.geometry import LineString
import tempfile
import json
import glob
import tqdm
from prettytable import PrettyTable


class Bev_Projector:
    def __init__(self, side_range, fwd_range, height_range, res, lane_width_x, lane_width_y):
        self.side_range = side_range  
        self.fwd_range = fwd_range   
        self.height_range = height_range  
        self.res = res
        self.lane_width_x = lane_width_x  
        self.lane_width_y = lane_width_y 
        self.zx_xmax = int((self.side_range[1] - self.side_range[0]) / self.res) 
        self.zx_ymax = int((self.fwd_range[1] - self.fwd_range[0]) / self.res)  
        self.zy_xmax = self.zx_ymax 
        self.zy_ymax = int((self.height_range[1] - self.height_range[0]) / self.res)

    def proj_oneline_zx(self, one_lane):
        """
        :param one_lane: N*3,[[x,y,z],...]
        :return:
        """
        img = np.zeros([self.zx_ymax, self.zx_xmax], dtype=np.uint8)

        one_lane = np.array(one_lane)
        one_lane = one_lane[one_lane[:, 2] < 10]
        lane_x = one_lane[:, 0]
        lane_z = one_lane[:, 2]

        x_img = (lane_x / self.res).astype(np.int32)  # 
        y_img = (-lane_z / self.res).astype(np.int32)

        x_img += int(self.side_range[1] / self.res)   
        y_img += int(self.fwd_range[1] / self.res) 

        for i in range(y_img.shape[0]-1):
            cv2.line(img, (x_img[i], y_img[i]), (x_img[i+1], y_img[i+1]), 255, self.lane_width_x)
        return img


class LaneEval:
    @staticmethod
    def file_parser(gt_root_path, pred_root_path):
        gt_files_list = glob.glob(os.path.join(gt_root_path, '*', 'cam01', '*.json'))
        pred_files_list = glob.glob(os.path.join(pred_root_path, '*', 'cam01', '*.json'))
        gt_files_list.sort()
        pred_files_list.sort()
        assert(len(gt_files_list) == len(pred_files_list) and len(gt_files_list) > 0)
        return gt_files_list, pred_files_list

    @staticmethod
    def summarize(res):
        gt_all = 0.
        pred_all = 0.
        tp_all = 0.
        distance_mean = 0.
        for res_spec in res:
            gt_all += res_spec[0]
            pred_all += res_spec[1]
            tp_all += res_spec[2]
            distance_mean += res_spec[3]

        precision = tp_all / pred_all
        recall = tp_all / gt_all
        if precision + recall == 0:
            F_value = 0.
        else:
            F_value = 2 * precision * recall / (precision + recall)

        distance_mean /= tp_all + 1e-5
        return dict(
            F_value=F_value,
            precision=precision,
            recall=recall,
            distance_error=distance_mean,
        )

    def lane_evaluation(self, pred_root_path, gt_root_path, test_cfg):
        gt_files_list, pred_files_list = self.file_parser(gt_root_path, pred_root_path)
        score_l = int(test_cfg["score_l"] * 100)
        score_h = int(test_cfg["score_h"] * 100)
        score_step = int(test_cfg["score_step"] * 100)
        score_num = int((score_h - score_l) / score_step)
        test_cfg['score_num'] = score_num

        tempfile.tempdir = '/dev/shm'
        tmp_dir = tempfile.mkdtemp()
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)

        evaluate_results = evaluate_list(gt_files_list, pred_files_list, test_cfg)
        gt_all = evaluate_results['gt_all']
        pr_all = evaluate_results['pr_all']
        tp_all = evaluate_results['tp_all']
        distance_error = evaluate_results['distance_error']

        precision = tp_all / pr_all
        recall = tp_all / gt_all

        F_value = 2 * precision * recall / (precision + recall)

        distance_error /= tp_all + 1e-5

        pt = PrettyTable()
        title_file = f'evaluate by {__file__}'
        pt.title = f'{title_file}'
        pt.field_names = ['prob_thresh', 'F1', 'precision', 'recall', 'D error']
        for i in range(score_l, score_h, score_step):
            index = int((i - score_l) / score_step)
            pt.add_row([str(i / 100),
                        F_value[index],
                        precision[index],
                        recall[index],
                        distance_error[index]
                        ])
        print(pt)
        return pt

def evaluate_list(gt_path_list, pred_path_list, config):
    bev_projector = Bev_Projector(side_range=(config['side_range_l'], config['side_range_h']),
                                  fwd_range=(config['fwd_range_l'], config['fwd_range_h']),
                                  height_range=(config['height_range_l'], config['height_range_h']),
                                  res=config['res'], lane_width_x=config['lane_width_x'],
                                  lane_width_y=config['lane_width_y'])
    score_num = config["score_num"]
    iou_thresh = config['iou_thresh']
    distance_thresh = config['distance_thresh']

    score_l = int(config["score_l"] * 100)
    score_h = int(config["score_h"] * 100)
    score_step = int(config["score_step"] * 100)

    gt_all = np.zeros((score_num,), dtype=np.float32)
    pr_all = np.zeros((score_num,), dtype=np.float32)
    tp_all = np.zeros((score_num,), dtype=np.float32)
    distance_error = np.zeros((score_num,), dtype=np.float32)

    idx = 0
    for gt_path, pred_path in tqdm.tqdm(zip(gt_path_list, pred_path_list)):
        idx += 1
        leof = LaneEvalOneFile(gt_path, pred_path, bev_projector, iou_thresh, distance_thresh, score_l, score_h, score_step)
        gt_num, pr_num, tp_num, distance_tmp = leof.eval()
        gt_all += gt_num
        pr_all += pr_num
        tp_all += tp_num
        distance_error += distance_tmp
    json_out_data = {"gt_all": gt_all, "pr_all": pr_all,
                     "tp_all": tp_all, "distance_error": distance_error}
    return json_out_data


class LaneEvalOneFile:
    def __init__(self, gt_path, pred_path, bev_projector, iou_thresh, distance_thresh, score_l, score_h, score_step):
        self.gt_path = gt_path
        self.pred_path = pred_path
        self.bev_projector = bev_projector
        self.iou_thresh = iou_thresh
        self.distance_thresh = distance_thresh
        self.score_l = score_l
        self.score_h = score_h
        self.score_step = score_step

    def preprocess(self, store_spec):
        gt_json = load_json(self.gt_path)
        pred_json = load_json(self.pred_path)
        gt_lanes3d = gt_json['lanes']
        gt_lanes3d = [gt_lanespec3d for gt_lanespec3d in gt_lanes3d if len(gt_lanespec3d) >= 2]
        gt_num = len(gt_lanes3d)
        pred_lanes3d = pred_json['lanes']
        pred_lanes3d = [pred_lanespec3d["points"] for pred_lanespec3d in pred_lanes3d if len(pred_lanespec3d) >= 2
                        and np.float(pred_lanespec3d["score"]) > store_spec]
        pred_num = len(pred_lanes3d)
        return gt_lanes3d, gt_num, pred_lanes3d, pred_num

    def calc_iou(self, lane1, lane2):
        """
        :param lane1:
        :param lane2:
        :return:
        """
        img1, img2 = self.bev_projector.proj_oneline_zx(lane1), self.bev_projector.proj_oneline_zx(lane2)

        union_im = cv2.bitwise_or(img1, img2)
        union_sum = union_im.sum()
        inter_sum = img1.sum() + img2.sum() - union_sum
        if union_sum == 0:
            return 0
        else:
            return inter_sum / float(union_sum)

    def cal_mean_dist(self, src_line, dst_line):
        """
        :param src_line: gt
        :param dst_line: pred
        :return:
        """
        src_line = LineString(np.array(src_line))
        dst_line = LineString(np.array(dst_line))

        total_distance = 0
        samples = np.arange(0.05, 1, 0.1)
        for sample in samples:
            total_distance += src_line.interpolate(sample, normalized=True).distance(dst_line)
        mean_distance = total_distance / samples.shape[0]
        return mean_distance

    def sort_lanes_z(self, lanes):
        sorted_lanes = list()
        for lane_spec in lanes:
            if lane_spec[0][-1] > lane_spec[1][-1]:
                lane_spec = lane_spec[::-1]
            sorted_lanes.append(lane_spec)
        return sorted_lanes

    def eval(self):
        gt_num = list()
        pred_num = list()
        tp = list()
        distance_error = list()
        for store in range(self.score_l, self.score_h, self.score_step):  # [0.1, 1, 0.05]
            store_spec = store * 0.01
            gt_lanes, gt_num_spec, pred_lanes, pred_num_spec = self.preprocess(store_spec)
            gt_lanes = self.sort_lanes_z(gt_lanes)
            pred_lanes = self.sort_lanes_z(pred_lanes)
            tp_spec, distance_error_spec = self.cal_tp(gt_num_spec, pred_num_spec, gt_lanes, pred_lanes)
            gt_num.append(gt_num_spec)
            pred_num.append(pred_num_spec)
            tp.append(tp_spec)
            distance_error.append(distance_error_spec)
        return gt_num, pred_num, tp, distance_error

    def cal_tp(self, gt_num, pred_num, gt_lanes, pred_lanes):
        tp = 0
        distance_error = 0
        if gt_num > 0 and pred_num > 0:
            iou_mat = [[0 for col in range(pred_num)] for row in range(gt_num)]
            for i in range(gt_num):
                for j in range(pred_num):
                    iou_mat[i][j] = self.calc_iou(gt_lanes[i], pred_lanes[j])
            cost_mat = []
            for row in iou_mat:
                cost_row = list()
                for col in row:
                    cost_row.append(1.0 - col)
                cost_mat.append(cost_row)
            m = Munkres()
            match_idx = m.compute(cost_mat)  #
            for row, col in match_idx:
                gt_lane = gt_lanes[row]
                pred_lane = pred_lanes[col]
                cur_distance = self.cal_mean_dist(gt_lane, pred_lane)
                if cur_distance < self.distance_thresh:
                    distance_error += cur_distance
                    tp += 1
        return tp, distance_error
