# import numpy as np
# import matplotlib.pyplot as plt
# import open3d as o3d
# fig = plt.figure() #创建一个绘图对象
# ax = fig.add_subplot(111, projection='3d')#111表示1行1列的第一个，即第一幅图
# #第二副图 与第一幅相隔

# ax1 = fig.add_subplot(122,projection='3d')
# plt.show()

# 点云分割
import numpy as np
import open3d as o3d
from sklearn.cluster import KMeans, DBSCAN, OPTICS
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
 
# Read point cloud:
pcd = o3d.io.read_point_cloud("road.pcd")
# Get points and transform it to a numpy array:
points = np.asarray(pcd.points).copy()

# Normalisation: 
scaled_points = StandardScaler().fit_transform(points) #标准化

# Clustering:
model = DBSCAN(eps=0.15, min_samples=10) #eps:半径 min_samples:最小样本数
# model = KMeans(n_clusters=4)
model.fit(scaled_points) #训练
# Get labels:
labels = model.labels_
# Get the number of colors:
n_clusters = len(set(labels)) 
 
# Mapping the labels classes to a color map:
colors = plt.get_cmap("tab20")(labels / (n_clusters if n_clusters > 0 else 1))
# Attribute to noise the black color:
colors[labels < 0] = 0
# Update points colors:
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
 
# Display:
o3d.visualization.draw_geometries([pcd])