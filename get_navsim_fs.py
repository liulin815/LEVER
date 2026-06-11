import pickle
import os
import lzma
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# 读取 .pkl 文
root_path = '/data/liulin/workspace/DiffusionDrive/exp/metric_cache'
traj_path = '/data/liulin/workspace/DiffusionDrive/trajs/traj.npy'
token_path = '/data/liulin/workspace/DiffusionDrive/trajs/token.pkl'
selected_token_path = '/data/liulin/workspace/DiffusionDrive/trajs/selected_token.pkl'
'''
def list_all_files(root_dir):
    """递归列出目录下所有文件（绝对路径）"""
    all_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            full_path = os.path.join(root, file)
            all_files.append(full_path)
    return all_files

def world_to_vehicle_transform(x, y, yaw):
    """
    构建世界坐标系 -> 自车坐标系的变换矩阵 (SE(2))
    :param x: 自车在世界坐标系的x位置
    :param y: 自车在世界坐标系的y位置
    :param yaw: 自车的偏航角（弧度）
    :return: 3x3 变换矩阵 T_{W -> V}
    """
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    
    # 旋转部分
    R = np.array([
        [cos_yaw, sin_yaw],
        [-sin_yaw, cos_yaw]
    ])
    
    # 平移部分
    t = np.array([-x, -y])
    t_rotated = R @ t  # 先旋转再平移
    
    # 组合成齐次变换矩阵
    T = np.eye(3)
    T[:2, :2] = R
    T[:2, 2] = t_rotated
    
    return T

def transform_point(T, px, py):
    """将世界坐标系的点 (px, py) 转换到自车坐标系"""
    p_world = np.array([px, py, 1])  # 齐次坐标
    p_vehicle = T @ p_world
    return p_vehicle[:2]  # 返回 (x', y')

files = list_all_files(root_path)
#import pdb;pdb.set_trace()
sample_tokens = []
selected_sample_tokens = []
trajs = []
i_count = 0
for file in files:
    try:
        with lzma.open(file, "rb") as f:
            data = pickle.load(f)
            sample_tokens.append(file.split('/')[-2])
            i_count = i_count + 1
    except:
        continue
    x = data.trajectory._trajectory[0].car_footprint.center.x
    y = data.trajectory._trajectory[0].car_footprint.center.y
    yaw = data.trajectory._trajectory[0].car_footprint.center.heading
    T = world_to_vehicle_transform(x, y, yaw)
    traj = []
    traj.append(np.array([0,0]))
    for i in range(1,51):
        x_i = data.trajectory._trajectory[i].car_footprint.center.x
        y_i = data.trajectory._trajectory[i].car_footprint.center.y
        traj.append(transform_point(T, x_i, y_i))
    trajs.append(np.stack(traj))
    print(i_count)
import pdb;pdb.set_trace()
with open(token_path, "wb") as f:
    pickle.dump(sample_tokens, f)
import pdb;pdb.set_trace()
np.save(traj_path, np.stack(trajs))
import pdb;pdb.set_trace()
print(data)
# p data.trajectory._trajectory[0].car_footprint.center.heading
'''
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
    
    for cluster_label, color in zip(unique_labels, colors):
        if cluster_label == -1 or cluster_label == 0 or cluster_label == 1:
            continue  # 跳过噪声簇
        
        # 获取当前簇的所有索引
        cluster_indices = [idx for idx, label in enumerate(labels) if label == cluster_label]
        
        # 随机抽取一个样本
        sample_idx = np.random.choice(cluster_indices)
        sample_traj = traj_array[sample_idx]
        
        # 绘制轨迹
        x = sample_traj[:, 0]
        y = sample_traj[:, 1]
        plt.plot(x, y, 
                markersize=10, 
                linewidth=2.5,
                marker='o', 
                color=color
        )#, label=f'Cluster {cluster_label}')
    
    plt.title("Sampled Trajectories from Each Cluster")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    #plt.grid(True)
    #plt.xticks(np.arange(-1.8, 2.5, 0.1))  # x轴从0到10，步长0.5
    #plt.yticks(np.arange(0, 40, 1))  # y轴从-1到1，步长0.2
    plt.savefig("/data/liulin/workspace/DiffusionDrive/vis1.png", bbox_inches='tight', dpi=600)
    #plt.show()

def cluster_trajectories1(traj_array, min_samples=5, percentile=53):
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


trajs = np.load(traj_path)
with open(token_path, "rb") as f:
    token_list = pickle.load(f)
#clustered_indices = cluster_trajectories(trajs)
trajs = trajs[:,:8,:]
trajs = trajs[..., ::-1] 
sorted_clusters, labels = cluster_trajectories1(trajs)
plot_sample_trajectories(trajs, sorted_clusters, labels)
import pdb;pdb.set_trace()
selected_sample_tokens = []
for i, indices in enumerate(clustered_indices):
    print(f"簇{i}：{len(indices)}个样本")
    if i == 0 or i == 1:
        continue
    for i_idx in indices:
        selected_sample_tokens.append(token_list[i_idx])
with open(selected_token_path, "wb") as f:
    pickle.dump(selected_sample_tokens, f)
#selected_token_path
import pdb;pdb.set_trace()
