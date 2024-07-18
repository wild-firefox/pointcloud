# 导入库
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 创建数据
A = np.array([[1, 2, 3], [3, 10, 1], [2, 8, 9], [1, 11, 12]])
y = A[:, 0] # 因变量 #A[:,0]表示取所有行的第0列 [[1,3,2,1]]
X = A[:, 1:] # 自变量 #A[:,1:]表示取所有行的第1列及其后面的列 [[2,3],[1,1],[8,9],[11,12]]

# 建立模型
model = LinearRegression() # 创建线性回归模型
model.fit(X, y)

# 绘制散点图
fig = plt.figure() # 创建画布
ax = fig.add_subplot(111, projection='3d') # 创建3D坐标轴 #111表示1行1列第1个子图 #projection='3d'表示3D坐标轴
ax.scatter(X[:, 0], X[:, 1], y, c='b', marker='o') # 绘制散点图 #X[:,0]表示取所有行的第0列 #X[:,1]表示取所有行的第1列 #y表示因变量 #c='b'表示颜色为蓝色 #marker='o'表示散点图的点为圆形

# 绘制拟合线
x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 10) # 生成x1坐标 #np.linspace()函数用于创建等差数列 #X[:,0].min()表示取所有行的第0列的最小值 #X[:,0].max()表示取所有行的第0列的最大值 #10表示生成10个数
x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 10) 
x1, x2 = np.meshgrid(x1, x2) # 生成网格点坐标矩阵 #np.meshgrid()函数用于生成网格点坐标矩阵 #x1表示x轴坐标矩阵 #x2表示y轴坐标矩阵
#print(model.predict(np.c_[x1.ravel(), x2.ravel()])) 100X1    .reshape(x1.shape) 变为10X10
y_pred = model.predict(np.c_[x1.ravel(), x2.ravel()]).reshape(x1.shape) # 生成拟合平面的z轴坐标 #np.c_[]函数用于按列连接两个矩阵 #x1.ravel()表示将x1转换为一维数组 #x2.ravel()表示将x2转换为一维数组 #np.c_[x1.ravel(), x2.ravel()]表示将x1和x2按列连接 #model.predict()函数用于预测 #np.c_[x1.ravel(), x2.ravel()]表示将x1和x2按列连接 #.reshape(x1.shape)表示将数组转换为x1的形状
ax.plot_surface(x1, x2, y_pred, color='r', alpha=0.5) # 绘制拟合平面 #ax.plot_surface()函数用于绘制三维曲面 #x1表示x轴坐标矩阵 #x2表示y轴坐标矩阵 #y_pred表示z轴坐标矩阵 #color='r'表示颜色为红色 #alpha=0.5表示透明度为0.5

# 设置坐标轴标签
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')

# 显示图形
plt.show()

# 查看结果
print('回归系数:', model.coef_)
print('截距项:', model.intercept_)
print('决定系数:', model.score(X, y))