import pickle
import numpy as np
from pyquaternion import Quaternion

def load_pkl_file(file_path):
    """
    读取PKL文件并返回其中的数据
    
    参数:
        file_path (str): PKL文件的路径
        
    返回:
        存储在PKL文件中的数据
    """
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到")
        return None
    except Exception as e:
        print(f"读取PKL文件时发生错误: {e}")
        return None

# 使用示例

def transform_to_target_vehicle(ego2global_source, ego2global, 
                                ego_pose_world, ego_pose_world_source, 
                                target_pose_ego, target_pose):
    """
    ego_pose_world: 自车在世界坐标系中的位姿 (x, y, z, yaw)
    target_pose_ego: 其他车辆在自车坐标系中的位姿 (x, y, z, yaw_relative)
    """
    x_ego_w, y_ego_w, z_ego_w, yaw_ego_w = ego_pose_world # t时刻自车世界坐标系
    yaw_ego_w_source = ego_pose_world_source[3] # 起始时刻自车偏航角度
    x_t_e, y_t_e, z_t_e, yaw_t_e = target_pose_ego # 汽车t时刻自车坐标系位置
    
    x_t, y_t, z_t, yaw_t = target_pose # 起始时刻目标位置
    p_target_ego_homo_source = np.array([x_t, y_t, z_t, 1])  # 齐次坐标
    p_target_world_homo_source = ego2global_source @ p_target_ego_homo_source # 世界坐标系下的起始时刻他车位置
    
    
    
    T_ego_to_world = ego2global
    p_target_ego_homo = np.array([x_t_e, y_t_e, z_t_e, 1])  # 齐次坐标
    p_target_world_homo = T_ego_to_world @ p_target_ego_homo
    
    x_t_w, y_t_w, z_t_w = p_target_world_homo_source[:3]
    yaw_t_w = yaw_ego_w_source + yaw_t  # 世界坐标系下的偏航角

    # 3. 构建世界到目标车的变换矩阵 T_world_to_target
    cos_t = np.cos(yaw_t_w)
    sin_t = np.sin(yaw_t_w)
    T_world_to_target = np.array([
        [cos_t, sin_t, 0, -x_t_w * cos_t - y_t_w * sin_t],
        [-sin_t, cos_t, 0, x_t_w * sin_t - y_t_w * cos_t],
        [0, 0, 1, -z_t_w],
        [0, 0, 0, 1]
    ])

    # 4. 转换坐标: 世界坐标系 → 目标车坐标系
    p_target_homo = T_world_to_target @ p_target_world_homo
    p_target = p_target_homo[:3]  # 非齐次坐标
    ego_motion = (ego_pose_world - ego_pose_world_source)[0:2]

    return p_target, ego_motion

def get_motion_data(idx,loaded_data):
    motion_list = []
    ego_list = []
    motion_list.append(np.array([0,0]))
    length = 8
    ins_id = 8
    ego2global_translation_source = loaded_data[idx]['ego2global_translation']
    ego_quaternion_source = Quaternion(loaded_data[idx]["ego2global_rotation"])
    global_ego_pose_source = np.array(
        [ego2global_translation_source[0], ego2global_translation_source[1], ego2global_translation_source[2], ego_quaternion_source.yaw_pitch_roll[0]],
        dtype=np.float64,
    )
    ego2global_source = loaded_data[idx]['ego2global']
    target_pose = np.array([loaded_data[idx]['anns']['gt_boxes'][ins_id][0],
                            loaded_data[idx]['anns']['gt_boxes'][ins_id][1],
                            loaded_data[idx]['anns']['gt_boxes'][ins_id][2],
                            loaded_data[idx]['anns']['gt_boxes'][ins_id][6]],
                        dtype=np.float64,)
    for i in range(idx,idx+length):
        ego2global_translation = loaded_data[i]['ego2global_translation']
        ego_quaternion = Quaternion(loaded_data[i]["ego2global_rotation"])
        global_ego_pose = np.array(
            [ego2global_translation[0], ego2global_translation[1], ego2global_translation[2], ego_quaternion.yaw_pitch_roll[0]],
            dtype=np.float64,
        )
        ego2global = loaded_data[i]['ego2global']
        target_pose_ego = np.array(
            [loaded_data[i]['anns']['gt_boxes'][ins_id][0],
            loaded_data[i]['anns']['gt_boxes'][ins_id][1],
            loaded_data[i]['anns']['gt_boxes'][ins_id][2],
            loaded_data[i]['anns']['gt_boxes'][ins_id][6]],
        dtype=np.float64)
        motion_point, ego_point = transform_to_target_vehicle(
            ego2global_source, ego2global, 
            global_ego_pose, global_ego_pose_source, 
            target_pose_ego, target_pose)
        motion_list.append(motion_point[0:2])
        ego_list.append(ego_point)
    return motion_list, ego_list


if __name__ == "__main__":
    pkl_file = "/data/liulin/workspace/DiffusionDrive/dataset/navsim_logs/trainval/2021.05.12.19.36.12_veh-35_00005_00204.pkl"  # 替换为你的PKL文件路径
    loaded_data = load_pkl_file(pkl_file)
    import pdb;pdb.set_trace()
    motion_list, ego_list = get_motion_data(100, loaded_data)
    import pdb;pdb.set_trace()
    if loaded_data is not None:
        print("PKL文件内容:")
        print(loaded_data)