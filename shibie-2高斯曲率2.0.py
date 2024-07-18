import open3d as o3d
import numpy as np
from scipy.optimize import curve_fit

voxel_size = 1 #0.01越小密度越大
radius = 5#500#0.5   #搜索半径
num_knn = 100 #搜索邻域点的个数 #越多
pcd = o3d.io.read_point_cloud("output_751_down_z.ply") # "flaw.pcd" "cropped_2.ply"
print(len(pcd.points))
#pcd = pcd.voxel_down_sample(voxel_size) #下采样
o3d.visualization.draw_geometries([pcd])
print(len(pcd.points))
cloud = pcd
points = np.asarray(cloud.points) #点云转换为数组 点云数组形式为[[x1,y1,z1],[x2,y2,z2],...]
kdtree = o3d.geometry.KDTreeFlann(cloud) #建立KDTree
num_points = len(cloud.points) #点云中点的个数
pcd.paint_uniform_color([1, 0, 0])  # 初始化所有颜色为红色
# 定义非线性函数，这里假设是一个二次曲面
def func(x, a, b, c, d, e, f):
    return a * x[0]**2 + b * x[1]**2 + c * x[0] * x[1] + d * x[0] + e * x[1] + f
def f(x, y):
    return popt[0]*x**2 +popt[1]*y**2 +popt[2]* x*y +popt[3]*x + popt[4]*y +popt[5]
# 定义曲面的梯度函数，即一阶偏导数
def gradient(f, x, y):
    # 使用中心差分法近似求导  #参考https://cloud.tencent.com/developer/article/1685164
    h = 1e-6 # 差分步长，可以根据精度要求调整
    df_dx = (f(x + h, y) - f(x - h, y)) / (2 * h) # 对x求偏导
    df_dy = (f(x, y + h) - f(x, y - h)) / (2 * h) # 对y求偏导
    return df_dx, df_dy
# 定义曲面的曲率函数，即二阶偏导数
def curvature(f, x, y):
    # 使用中心差分法近似求导
    h = 1e-3 # 差分步长，可以根据精度要求调整 #越小越精确
    d2f_dx2 = (f(x + h, y) - 2 * f(x, y) + f(x - h, y)) / (h ** 2) # 对x求二阶偏导
    d2f_dy2 = (f(x, y + h) - 2 * f(x, y) + f(x, y - h)) / (h ** 2) # 对y求二阶偏导
    d2f_dxdy = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ** 2) # 对xy求混合偏导
    # 根据公式计算高斯曲率K和平均曲率H
    df_dx, df_dy = gradient(f, x, y) # 调用梯度函数求一阶偏导
    E = 1 + df_dx ** 2
    F = df_dx * df_dy
    G = 1 + df_dy ** 2
    L = d2f_dx2 / np.sqrt(1 + df_dx ** 2 + df_dy ** 2) #np.sqrt()表示开方
    M = d2f_dxdy / np.sqrt(1 + df_dx ** 2 + df_dy ** 2)
    N = d2f_dy2 / np.sqrt(1 + df_dx ** 2 + df_dy ** 2)
    K = (L * N - M ** 2) / (E * G - F ** 2) # 高斯曲率
    H = (E * N + G * L - 2 * F * M) / (2 * (E * G - F ** 2)) # 平均曲率
    return K, H

curvatures = []  
for i in range(num_points):
    #k, idx, _ = kdtree.search_radius_vector_3d(cloud.points[i], radius) #返回邻域点的个数和索引
    k, idx, _ = kdtree.search_knn_vector_3d(cloud.points[i], num_knn) 
    #k, idx, _ = kdtree.search_hybrid_vector_3d(cloud.points[i],radius, num_knn) #返回半径内不超过num_knn个点的个数和索引
    neighbors = points[idx] #数组形式为[[x1,y1,z1],[x2,y2,z2],...]
    #print(k)
    Y = neighbors[:, 2] # 因变量
    X = neighbors[:, [0,1]] # 自变量 [[2,3],[1,1],[8,9],[11,12],[4,5],[8,9]] 6*2
    popt, pcov = curve_fit(func, xdata=X.T,ydata= Y)
    x = cloud.points[i][0] # 某一点的x坐标
    y = cloud.points[i][1] # 某一点的y坐标
    K, H = curvature(f, x, y) # 计算该点的曲率
    curvatures.append([K,H])

print(len(curvatures))
print(curvatures)
limit_max = 1e-3#1e-3#0#1e-8 ##1e-3
for i in range(len(curvatures)):
    if -limit_max<=curvatures[i][0] <= limit_max and -limit_max<=curvatures[i][1] <=limit_max: #平坦
        np.asarray(pcd.colors)[i] = [0, 0, 0]#黑
    elif -limit_max<=curvatures[i][0] <= limit_max and curvatures[i][1] >limit_max:  #凸
        np.asarray(pcd.colors)[i] = [1, 0, 0]#红
    elif -limit_max<=curvatures[i][0] <= limit_max and -limit_max<curvatures[i][1] <limit_max: #凹
        np.asarray(pcd.colors)[i] = [0, 1, 0]#绿
    elif curvatures[i][0] < -limit_max and curvatures[i][1] >limit_max: #鞍形脊 大部分凸，少部分凹
        np.asarray(pcd.colors)[i] = [0, 0, 1]#蓝
    elif curvatures[i][0] < -limit_max and curvatures[i][1] <-limit_max: #鞍形谷 大部分凹，少部分凸
        np.asarray(pcd.colors)[i] = [0, 1, 1]#青
    elif curvatures[i][0] > limit_max and curvatures[i][1] >limit_max: #峰
        np.asarray(pcd.colors)[i] = [1, 0, 1]#紫
    elif curvatures[i][0] > limit_max and curvatures[i][1] <-limit_max: #阱
        np.asarray(pcd.colors)[i] = [1, 1, 0]#黄


#显示点云
o3d.visualization.draw_geometries([pcd])

#保存点云
o3d.io.write_point_cloud("output_v2_曲率分类_.pcd", pcd)


 
