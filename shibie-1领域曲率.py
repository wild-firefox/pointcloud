import open3d as o3d
import numpy as np




 
def pca_compute(data, sort=True):
    """
    点云的特征值与特征向量
    :param data: 点云数据 numpy数组形式为[[x1,y1,z1],[x2,y2,z2],...]
    :return: 特征值
    """
    average_data = np.mean(data, axis=0) # 求每一列的平均值，即求各个特征的平均值
    decentration_matrix = data - average_data  # 去中心化矩阵
 
    H = np.dot(decentration_matrix.T, decentration_matrix)  # 求协方差矩阵 #协方差是衡量两个变量关系的统计量，协方差为正表示两个变量正相关，为负表示两个变量负相关
    eigenvectors, eigenvalues, eigenvectors_T = np.linalg.svd(H) # 求特征值与特征向量 #H = UΣV^T #输出列向量、对角矩阵、行向量
    if sort:
        sort = eigenvalues.argsort()[::-1] # 从大到小排序 .argsort()是升序排序，[::-1]是将数组反转，实现降序排序
        eigenvalues = eigenvalues[sort] # 特征值  ## 使用索引来获取排序后的数组
    return eigenvalues
 
def caculate_surface_curvature(radius,pcd):
    """
    计算点云的表面曲率
    :param radius: 邻域半径
    :return: 点云中每个点的表面曲率
    """
    cloud = pcd
    points = np.asarray(cloud.points) #点云转换为数组 点云数组形式为[[x1,y1,z1],[x2,y2,z2],...]
    kdtree = o3d.geometry.KDTreeFlann(cloud) #建立KDTree
    num_points = len(cloud.points) #点云中点的个数
 
    curvature = []  # 储存表面曲率
    for i in range(num_points):
        k, idx, _ = kdtree.search_radius_vector_3d(cloud.points[i], radius) #返回邻域点的个数和索引
 
        neighbors = points[idx] #数组形式为[[x1,y1,z1],[x2,y2,z2],...]
        w = pca_compute(neighbors)  #由降序排序，w[2]为最小特征值  #np.zeros_like(w[2])生成与w[2]相同形状的全0数组
        delt = np.divide(w[2], np.sum(w), out=np.zeros_like(w[2]), where=np.sum(w) != 0) #根据公式求取领域曲率
        curvature.append(delt)
    curvature = np.array(curvature, dtype=np.float64)
 
    return curvature
 
def curvature_normal():
    '''
    传入的曲率curvature归一化，映射到0-1之间
    :return:
    '''
    data_normal = caculate_surface_curvature(radius,pcd)
    data_max = max(data_normal)
    data_min = min(data_normal)
    cur_normal = [(float(i) - data_min) / (data_max - data_min) for i in data_normal] 
    return cur_normal
 
def draw(cur_max,cur_min,pcd):
    '''
    绘图法向量绘图，曲率可视化绘图
    :param cur_max:表面曲率最大值
    :param cur_min:表面曲率最小值
    :return:
    '''
    cur_normal = curvature_normal()
    downpcd_normals = pcd
    print(pcd)
    print(cur_normal)
 
    pcd.paint_uniform_color([0.5,0.5,0.5])
    for i in range(len(cur_normal)):
        if 0 < cur_normal[i] <= cur_min: #归一化后的曲率
            np.asarray(pcd.colors)[i] = [1, 0, 0]#红
        elif cur_min < cur_normal[i] <= cur_max:
            np.asarray(pcd.colors)[i] = [0, 1, 0]#绿
        elif cur_max < cur_normal[i] <= 1: 
            np.asarray(pcd.colors)[i] = [0, 0, 1]#蓝
 
    # 可视化
    o3d.visualization.draw_geometries([downpcd_normals],window_name="可视化原始点云",
                                      width=800, height=800, left=50, top=50,
                                      mesh_show_back_face=False)
    return None


cur_max = 0.7 
cur_min = 0.4 #曲率分割基准
radius = 0.05
voxel_size = 0.01 #越小密度越大
pcd = o3d.io.read_point_cloud("bunny.pcd")
pcd = pcd.voxel_down_sample(voxel_size) #下采样
print(pcd)
curvature=caculate_surface_curvature(radius,pcd)
print(curvature)
draw(cur_max,cur_min,pcd)
 
 
