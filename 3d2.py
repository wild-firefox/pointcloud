# 导入库
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 创建数据
#A = np.array([[1, 2, 3], [3, 1, 1], [2, 8, 9], [1, 11, 12],[3,4,5],[7,8,9]])
# 生成随机数据
np.random.seed(0) # 设置随机种子
n = 100 # 数据点个数
x = np.linspace(0, 10, n) # x坐标
y = np.random.randint(1, 10, n) # y坐标
z = 2 * y**2 + 3 + np.random.normal(0, 1, size=n) # y坐标
A = np.vstack([x,y,z]).T #按垂直方向（行顺序）堆叠数组构成一个新的数组 成3x100的矩阵 然后转置成100x3的矩阵

y = A[:, 2] # 因变量
X = A[:, [0,1]] # 自变量 [[2,3],[1,1],[8,9],[11,12],[4,5],[8,9]] 6*2

# 定义非线性函数，这里假设是一个二次曲面
def func(x, a, b, c, d, e, f):
    return a * x[0]**2 + b * x[1]**2 + c * x[0] * x[1] + d * x[0] + e * x[1] + f

# 拟合模型，得到回归参数和参数协方差
#popt, pcov = curve_fit(func, X.T, y)
popt, pcov, infodict, mesg, ier = curve_fit (func, xdata=X.T,ydata= y, full_output = True)
#X.T是X的转置 是为了符合func函数的参数形式 (2*6)  xdata 可以是多维数组 K*n
#ydata 只能是一维数组 长度为n

# 打印回归参数
print('回归参数:', popt)
print('参数协方差:', pcov)
print('infodict:', infodict) #infodict
print('mesg:', mesg)
print('ier:', ier)


# 绘制散点图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y, c='b', marker='o')

# 绘制拟合曲面
x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 10)
x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 10)
x1, x2 = np.meshgrid(x1, x2)
y_pred = func([x1, x2], *popt)
ax.plot_surface(x1, x2, y_pred, color='r', alpha=0.5)

# 设置坐标轴标签
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')

# 显示图形
plt.show()