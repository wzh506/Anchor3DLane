import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import List, Dict, Tuple
import math
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置为黑体
matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号

class MTMAttentionVisualizer:
    def __init__(self, grid_size=8, num_objects=3, random_seed=None):
        self.grid_size = grid_size
        self.num_objects = num_objects
        self.current_frame = 0
        self.fig = None
        self.axes = None
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
            import random
            random.seed(random_seed)

    def generate_frame_data(self, frame_index: int) -> List[Dict]:
        """生成多样化的目标检测数据，支持可复现和多样性"""
        objects = []
        # 让类别、速度、初始位置等更随机
        for i in range(self.num_objects):
            # 随机类别
            category = np.random.choice([0, 1, 2])  # 支持更多类别
            # 随机初始位置和速度
            base_x = np.random.uniform(0, self.grid_size-2) + math.sin(frame_index * 0.2 + i) * np.random.uniform(0.5, 1.2)
            base_y = np.random.uniform(0, self.grid_size-2) + math.cos(frame_index * 0.3 + i) * np.random.uniform(0.5, 1.2)
            velocity = np.random.uniform(-0.5, 0.5, size=2) + [math.sin(frame_index * 0.1 + i) * 0.3, math.cos(frame_index * 0.1 + i) * 0.3]
            confidence = 0.7 + np.random.random() * 0.3
            objects.append({
                'id': i,
                'center': np.array([base_x, base_y]),
                'velocity': np.array(velocity),
                'category': category,
                'confidence': confidence
            })
        return objects
    
    def align_centers(self, prev_objects: List[Dict], delta_t: float = 0.1) -> List[Dict]:
        """根据速度对齐前一帧的目标中心"""
        aligned_objects = []
        for obj in prev_objects:
            aligned_obj = obj.copy()
            aligned_obj['aligned_center'] = obj['center'] + obj['velocity'] * delta_t
            aligned_objects.append(aligned_obj)
        return aligned_objects
    
    def l2_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """计算L2距离"""
        return np.linalg.norm(pos1 - pos2)
    
    def generate_cost_matrix(self, current_objects: List[Dict], 
                           aligned_prev_objects: List[Dict], 
                           threshold: float = 2.0) -> np.ndarray:
        """生成成本矩阵"""
        Nt = len(current_objects)
        Nt_1 = len(aligned_prev_objects)
        cost_matrix = np.zeros((Nt, Nt_1))
        
        for i in range(Nt):
            for j in range(Nt_1):
                distance = self.l2_distance(current_objects[i]['center'], 
                                          aligned_prev_objects[j]['aligned_center'])
                
                # 应用掩码：不同类别或距离过大的设为无穷大
                if (distance > threshold or 
                    current_objects[i]['category'] != aligned_prev_objects[j]['category']):
                    cost_matrix[i, j] = 1e8
                else:
                    cost_matrix[i, j] = distance
        
        return cost_matrix
    
    def row_wise_softmax(self, matrix: np.ndarray) -> np.ndarray:
        """行级别的Softmax函数"""
        result = np.zeros_like(matrix)
        
        for i in range(matrix.shape[0]):
            row = matrix[i]
            # 过滤掉无穷大值
            valid_indices = row < 1e8
            if np.any(valid_indices):
                valid_values = row[valid_indices]
                max_val = np.max(valid_values)
                exp_values = np.exp(-(valid_values - max_val))
                sum_exp = np.sum(exp_values)
                
                result[i, valid_indices] = exp_values / sum_exp if sum_exp > 0 else 0
            else:
                result[i] = 0
        
        return result
    
    def generate_cross_attention(self, current_objects, prev_objects):
        """基于L2距离的传统交叉注意力（无掩码/阈值）"""
        Nt = len(current_objects)
        Nt_1 = len(prev_objects)
        attention = np.zeros((Nt, Nt_1))
        for i in range(Nt):
            for j in range(Nt_1):
                dist = self.l2_distance(current_objects[i]['center'], prev_objects[j]['aligned_center'])
                attention[i, j] = -dist  # 距离越近，注意力越大
        # 行softmax
        for i in range(Nt):
            row = attention[i]
            exp_row = np.exp(row - np.max(row))
            sum_exp = np.sum(exp_row)
            attention[i] = exp_row / sum_exp if sum_exp > 0 else 0
        # 填充到固定大小
        full_attention = np.zeros((self.grid_size, self.grid_size))
        full_attention[:Nt, :Nt_1] = attention
        return full_attention
    
    def get_current_attention_maps(self, frame_index: int) -> Dict:
        """计算当前帧的注意力图"""
        current_objects = self.generate_frame_data(frame_index)
        prev_objects = self.generate_frame_data(frame_index - 1)
        aligned_prev_objects = self.align_centers(prev_objects)
        
        # MTM注意力
        cost_matrix = self.generate_cost_matrix(current_objects, aligned_prev_objects)
        mtm_attention = self.row_wise_softmax(cost_matrix)
        
        # 填充到固定大小
        mtm_full = np.zeros((self.grid_size, self.grid_size))
        mtm_full[:mtm_attention.shape[0], :mtm_attention.shape[1]] = mtm_attention
        
        # 交叉注意力（真实实现）
        cross_attention = self.generate_cross_attention(current_objects, aligned_prev_objects)
        
        return {
            'mtm': mtm_full,
            'cross': cross_attention,
            'current_objects': current_objects,
            'prev_objects': aligned_prev_objects,
            'cost_matrix': cost_matrix
        }
    
    def plot_static_comparison(self, frame_index: int = 0):
        """绘制静态对比图+目标分布图"""
        data = self.get_current_attention_maps(frame_index)
        Nt = len(data['current_objects'])
        Nt_1 = len(data['prev_objects'])
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        # 1. 目标分布图（含位移箭头）
        ax0 = axes[0]
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        markers = ['o', 's', '^', 'D', 'P', 'X']
        # 当前帧目标
        for i, obj in enumerate(data['current_objects']):
            ax0.scatter(obj['center'][0], obj['center'][1], c=colors[obj['category']%len(colors)],
                        marker=markers[obj['category']%len(markers)], s=120, label=f'当前-类别{obj["category"]}' if i==0 else "")
        # 前一帧目标（对齐后）
        for i, obj in enumerate(data['prev_objects']):
            ax0.scatter(obj['aligned_center'][0], obj['aligned_center'][1], c=colors[obj['category']%len(colors)],
                        marker='x', s=120, label=f'前一帧-类别{obj["category"]}' if i==0 else "")
        # 画位移箭头（从前一帧对齐后位置指向当前帧位置）
        for i, obj in enumerate(data['current_objects']):
            # 找到类别相同且距离最近的前一帧目标
            min_dist = float('inf')
            match_j = -1
            for j, prev_obj in enumerate(data['prev_objects']):
                if obj['category'] == prev_obj['category']:
                    dist = np.linalg.norm(obj['center'] - prev_obj['aligned_center'])
                    if dist < min_dist:
                        min_dist = dist
                        match_j = j
            if match_j >= 0:
                prev_obj = data['prev_objects'][match_j]
                ax0.arrow(prev_obj['aligned_center'][0], prev_obj['aligned_center'][1],
                          obj['center'][0] - prev_obj['aligned_center'][0],
                          obj['center'][1] - prev_obj['aligned_center'][1],
                          color=colors[obj['category']%len(colors)],
                          width=0.05, head_width=0.25, length_includes_head=True, alpha=0.7)
        ax0.set_title('目标空间分布及位移', fontsize=12)
        ax0.set_xlabel('X')
        ax0.set_ylabel('Y')
        ax0.grid(True, alpha=0.3)
        ax0.set_xlim(0, self.grid_size)
        ax0.set_ylim(0, self.grid_size)
        handles, labels = ax0.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax0.legend(by_label.values(), by_label.keys(), loc='best')
        # 2. 交叉注意力
        im1 = axes[1].imshow(data['cross'][:Nt, :Nt_1], cmap='viridis', vmin=0, vmax=1)
        axes[1].set_title('(a) 交叉注意力', fontsize=12)
        axes[1].set_xlabel('前一帧目标 (t-1)')
        axes[1].set_ylabel('当前帧目标 (t)')
        # 3. MTM注意力
        im2 = axes[2].imshow(data['mtm'][:Nt, :Nt_1], cmap='viridis', vmin=0, vmax=1)
        axes[2].set_title('(b) MTM注意力', fontsize=12)
        axes[2].set_xlabel('前一帧目标 (t-1)')
        axes[2].set_ylabel('当前帧目标 (t)')
        # 颜色条
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(axes[2])
        cax = divider.append_axes("right", size="5%", pad=0.15)
        cbar = plt.colorbar(im2, cax=cax)
        cbar.set_label('注意力强度', rotation=270, labelpad=20, fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 1])
        plt.suptitle(f'MTM注意力机制热力图与目标分布 (Frame {frame_index})', fontsize=14, y=1.02)
        plt.show()
        # 打印详细信息
        self.print_frame_info(data, frame_index)
    
    
    

def main():
    """主函数"""
    print("MTM注意力机制热力图复现")
    print("=" * 50)
    # 创建可视化器，指定随机种子保证可复现
    visualizer = MTMAttentionVisualizer(grid_size=8, num_objects=5, random_seed=42)
    visualizer.plot_static_comparison(frame_index=5)
    
  

if __name__ == "__main__":
    main()