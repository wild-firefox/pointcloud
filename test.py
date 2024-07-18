import open3d as o3d
import numpy as np

#将ply转换成pcd
def convert_ply_to_pcd(ply_file, pcd_file):
    # 读取PLY文件
    point_cloud = o3d.io.read_point_cloud(ply_file)
 
    # 保存为PCD文件
    o3d.io.write_point_cloud(pcd_file, point_cloud)
 
# 使用示例
# ply_file_path = r'bunny\reconstruction\bun_zipper.ply'  #填入ply文件的路径
# pcd_file_path = r'bunny0.pcd'  #填入pcd文件的路径
# convert_ply_to_pcd(ply_file_path, pcd_file_path)

'''
pcd = o3d.io.read_point_cloud("bunny.pcd")
print(pcd)
pcd0 = o3d.io.read_point_cloud("bunny0.pcd")
print(pcd0)
o3d.visualization.draw_geometries([pcd,pcd0])
'''
'''
print("->正在加载点云... ")
pcd = o3d.io.read_point_cloud("bunny.pcd")
print(pcd)

# 法线估计
radius = 0.01   # 搜索半径
max_nn = 30     # 邻域内用于估算法线的最大点数
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))     # 执行法线估计

# 可视化
o3d.visualization.draw_geometries([pcd], 
                                  window_name = "可视化参数设置",
                                  width = 600,
                                  height = 450,
                                  left = 30,
                                  top = 30,
                                  point_show_normal = True)
'''

'''
import open3d as o3d
import numpy as np

##############li =[1,2,3,4,5,6,7,8,9,10]
print("->正在加载点云... ")
pcd = o3d.io.read_point_cloud("bunny.pcd")
print(pcd)

# 将点云设置为灰色
pcd.paint_uniform_color([0.5, 0.5, 0.5])

# 建立KDTree
pcd_tree = o3d.geometry.KDTreeFlann(pcd)

# 将第1500个点设置为紫色
pcd.colors[1500] = [0.5, 0, 0.5]
#############print(np.asarray(pcd.colors)[[li[1:]]])          np.asarray(pcd.colors)[idx_k[1:],:] 中,:可去
# 使用K近邻，将第1500个点最近的5000个点设置为蓝色
print("使用K近邻，将第1500个点最近的5000个点设置为蓝色")  
k = 5000    # 设置K的大小
[num_k, idx_k, _] = pcd_tree.search_knn_vector_3d(pcd.points[1500], k)    # 返回邻域点的个数和索引
np.asarray(pcd.colors)[idx_k[1:]] = [0, 0, 1]  # 跳过最近邻点（查询点本身）进行赋色
print("k邻域内的点数为：", num_k)

# 使用半径R近邻，将第1500个点半径（0.02）范围内的点设置为红色
print("使用半径R近邻，将第1500个点半径（0.02）范围内的点设置为红色")
radius = 0.02   # 设置半径大小
[num_radius, idx_radius, _] = pcd_tree.search_radius_vector_3d(pcd.points[1500], radius)   # 返回邻域点的个数和索引
np.asarray(pcd.colors)[idx_radius[1:]] = [1, 0, 0]  # 跳过最近邻点（查询点本身）进行赋色
print("半径r邻域内的点数为：", num_radius)

# 使用混合邻域，将半径R邻域内不超过max_num个点设置为绿色
print("使用混合邻域，将第1500个点半径R邻域内不超过max_num个点设置为绿色")
max_nn = 200   # 半径R邻域内最大点数
[num_hybrid, idx_hybrid, _] = pcd_tree.search_hybrid_vector_3d(pcd.points[1500], radius, max_nn)
np.asarray(pcd.colors)[idx_hybrid[1:]] = [0, 1, 0]  # 跳过最近邻点（查询点本身）进行赋色
print("混合邻域内的点数为：", num_hybrid)

print("->正在可视化点云...")
o3d.visualization.draw_geometries([pcd])
'''

'''

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

print("->正在加载点云... ")
pcd = o3d.io.read_point_cloud("road.pcd")
print(pcd)

print("->正在DBSCAN聚类...")
eps =    0.3    # 同一聚类中最大点间距 #eps要根据点云的大小来设置，太小会导致所有点都被聚类，太大会导致所有点都被聚为一类        
min_points = 50     # 有效聚类的最小点数
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(pcd.cluster_dbscan(eps, min_points, print_progress=True))
max_label = labels.max()    # 获取聚类标签的最大值 [-1,0,1,2,...,max_label]，label = -1 为噪声，因此总聚类个数为 max_label + 1
print(f"point cloud has {max_label + 1} clusters")
#colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
# 获取颜色映射对象
cmap = plt.get_cmap("tab20")
# 应用颜色映射对象
normalized_labels = labels / (max_label if max_label > 0 else 1) # 将标签归一化到[0,1]区间
colors = cmap(normalized_labels)
colors[labels < 0] = 0  # labels = -1 的簇为噪声，以黑色显示
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([pcd])
'''

import open3d as o3d

# 从PCD文件中读取点云
pcd = o3d.io.read_point_cloud("bunny.pcd")

# 估计点云的法线
pcd.estimate_normals()
# 使用Ball Pivoting算法从点云创建一个网格
radii = [0.005, 0.01, 0.02, 0.04]
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd, o3d.utility.DoubleVector(radii))

# 计算网格的顶点法线
mesh.compute_vertex_normals()

# 从网格中采样点
sampled_pcd = mesh.sample_points_poisson_disk(3000)

# 显示采样后的点云
print("Displaying sampled pointcloud ...")
o3d.visualization.draw_geometries([sampled_pcd])

# 显示重建的网格
print("Displaying reconstructed mesh ...")
o3d.visualization.draw_geometries([mesh])