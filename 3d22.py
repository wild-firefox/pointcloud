import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义二次曲面的函数模型
def func(p, x, y):
    a, b, c, d, e, f = p
    return a * x ** 2 + b * y ** 2 + c * x * y + d * x + e * y + f

# 定义误差函数
def error(p, x, y, z):
    return func(p, x, y) - z

# 数据点
A = np.array([[1, 2, 3], [3, 1, 1], [2, 8, 9], [1, 11, 12],[3,4,5],[7,8,9]])
x = A[:, 0]
y = A[:, 1]
z = A[:, 2]

# 初始参数
p0 = [1, 1, 1, 1, 1, 1]

# 拟合
plsq = leastsq(error, p0, args=(x,z,y))

# 输出结果
print("拟合参数:", plsq[0])
print("拟合误差:", plsq[1]) #plsq[1]表示损失函数的值

# 绘制散点图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='b', marker='o')

# 绘制拟合曲面
x = np.linspace(x.min(), x.max(), 10)
y = np.linspace(y.min(), y.max(), 10)
x, y = np.meshgrid(x, y)
z = func(plsq[0],x,y ) 
ax.plot_surface(x,y,z, color='r', alpha=0.5)

# 设置坐标轴标签
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# 显示图形
plt.show()