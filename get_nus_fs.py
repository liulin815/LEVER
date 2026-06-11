import os
import pickle
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from nuscenes.nuscenes import NuScenes
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

import mmcv

K = 6

def compute_adaptive_eps(data, min_samples=5, percentile=53):
    """自动计算DBSCAN的eps参数"""
    nn = NearestNeighbors(n_neighbors=min_samples)
    nn.fit(data)
    distances, _ = nn.kneighbors(data)
    k_distances = distances[:, -1]
    return np.percentile(k_distances, percentile)

def cluster_trajectories(traj_array, min_samples=5, percentile=53):
    """
    对轨迹数据进行自适应聚类，返回按簇大小排序的索引列表
    
    参数：
    traj_array: 形状为[N,6,2]的轨迹数组
    min_samples: DBSCAN的最小样本数参数
    percentile: 用于自动计算eps的百分位数
    
    返回：
    按簇大小降序排列的索引列表，每个元素为一个簇的索引数组
    """
    # 数据预处理：展平轨迹数据
    n_trajectories = traj_array.shape[0]
    flattened_data = traj_array.reshape(n_trajectories, -1)
    
    # 自动计算eps参数
    eps = compute_adaptive_eps(flattened_data, min_samples, percentile)
    
    # 使用DBSCAN进行聚类
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(flattened_data)
    
    # 统计簇信息并排序（包含噪声簇）
    unique_labels, counts = np.unique(labels, return_counts=True)
    clusters_info = sorted(zip(unique_labels, counts), 
                         key=lambda x: x[1], reverse=True)
    
    # 创建簇索引字典
    cluster_indices = {}
    for idx, label in enumerate(labels):
        cluster_indices.setdefault(label, []).append(idx)
    
    # 生成排序后的簇索引列表
    sorted_clusters = [cluster_indices[label] for label, _ in clusters_info]
    
    return sorted_clusters



def cluster_trajectories1(traj_array, min_samples=5, percentile=65):
    """
    对轨迹数据进行自适应聚类，返回按簇大小排序的索引列表
    
    参数：
    traj_array: 形状为[N,6,2]的轨迹数组
    min_samples: DBSCAN的最小样本数参数
    percentile: 用于自动计算eps的百分位数
    
    返回：
    按簇大小降序排列的索引列表，每个元素为一个簇的索引数组
    """
    # 数据预处理：展平轨迹数据
    n_trajectories = traj_array.shape[0]
    flattened_data = traj_array.reshape(n_trajectories, -1)
    
    # 自动计算eps参数
    eps = compute_adaptive_eps(flattened_data, min_samples, percentile)
    
    # 使用DBSCAN进行聚类
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(flattened_data)
    
    # 统计簇信息并排序（包含噪声簇）
    unique_labels, counts = np.unique(labels, return_counts=True)
    clusters_info = sorted(zip(unique_labels, counts), 
                         key=lambda x: x[1], reverse=True)
    
    # 创建簇索引字典
    cluster_indices = {}
    for idx, label in enumerate(labels):
        cluster_indices.setdefault(label, []).append(idx)
    
    # 生成排序后的簇索引列表
    sorted_clusters = [cluster_indices[label] for label, _ in clusters_info]
    
    return sorted_clusters, labels

def plot_sample_trajectories(traj_array, sorted_clusters, labels):
    """
    从每个簇中随机抽取一个样本并绘制其轨迹
    
    参数：
    traj_array: 形状为[N,6,2]的轨迹数组
    sorted_clusters: 按簇大小排序的索引列表
    labels: 每个样本的簇标签
    """
    plt.figure(figsize=(16, 12),dpi=600)
    
    # 为每个簇分配一个颜色
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    #import pdb;pdb.set_trace()
    for cluster_label, color in zip(unique_labels, colors):
        if cluster_label == 0: #or cluster_label == 1: #or cluster_label == 0 or cluster_label == 1:
        # 获取当前簇的所有索引
            #import pdb;pdb.set_trace()
            cluster_indices = [idx for idx, label in enumerate(labels) if label == cluster_label]
            
            # 随机抽取一个样本
            #sample_idx = np.random.choice(cluster_indices)
            for sample_idx in cluster_indices:
                #import pdb;pdb.set_trace()
                if sample_idx > 800:
                    continue
                color = colors[sample_idx%23]
                sample_traj = traj_array[sample_idx]
                #import pdb;pdb.set_trace()
                if abs(sample_traj[0][0]-sample_traj[-1][0]) < 1.5:
                    x = sample_traj[:, 0]
                    y = sample_traj[:, 1]
                    plt.plot(x, y, marker='o', color=color, alpha=0.5)#, label=f'Cluster {cluster_label}')
    
    plt.title("Sampled Trajectories from Each Cluster")
    plt.xlim(-2.5, 2.5)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    #plt.grid(True)
    plt.savefig("./vis1.png", bbox_inches='tight', dpi=600)
    #plt.show()


fp = './data/infos/nuscenes_infos_val.pkl'
data = mmcv.load(fp)
nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuscenes/', verbose=True)
data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
#import pdb;pdb.set_trace()
navi_trajs = [[], [], []]
fs_trajs = []
for idx in tqdm(range(len(data_infos))):
    info = data_infos[idx]
    #import pdb;pdb.set_trace()
    plan_traj = info['gt_ego_fut_trajs'].cumsum(axis=-2)
    fs_trajs.append(plan_traj)
fs_trajs = np.stack(fs_trajs)
clustered_indices = cluster_trajectories(fs_trajs)
#sorted_clusters, labels = cluster_trajectories1(fs_trajs)
#plot_sample_trajectories(fs_trajs, sorted_clusters, labels)
import pdb;pdb.set_trace()
scene_set = set()
scene_dict = {}
scene_tokens = []
sample_idxs = []
for i, indices in enumerate(clustered_indices):
    print(f"簇{i}：{len(indices)}个样本")
    if i == 0 or i == 1:
        continue
    for i_idx in indices:
        scene_name = nusc.get('scene', data_infos[i_idx]['scene_token'])['name']
        scene_set.add(scene_name)
        if scene_name not in scene_dict.keys():
            scene_dict[scene_name] = 1
        else:
            scene_dict[scene_name] = scene_dict[scene_name] + 1
        scene_tokens.append(i_idx)
        sample_idxs.append(data_infos[i_idx]['token'])
import pdb;pdb.set_trace()
print("hhhhhhhhhhh")


