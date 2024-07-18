'''
# 导入numpy库
import numpy as np

# 定义曲面函数z = f(x, y)
def f(x, y):
    return 0.1*(x)**2+0.1*x*y+0.1*y**2    #1 / (1 + np.exp(-x)) # 这里只是一个示例，您可以替换成您的曲面函数

# 定义曲面的梯度函数，即一阶偏导数
def gradient(f, x, y):
    # 使用中心差分法近似求导  #https://cloud.tencent.com/developer/article/1685164
    h = 1e-6 # 差分步长，可以根据精度要求调整
    df_dx = (f(x + h, y) - f(x - h, y)) / (2 * h) # 对x求偏导
    df_dy = (f(x, y + h) - f(x, y - h)) / (2 * h) # 对y求偏导
    return df_dx, df_dy

# 定义曲面的曲率函数，即二阶偏导数
def curvature(f, x, y):
    # 使用中心差分法近似求导
    h = 1e-6 # 差分步长，可以根据精度要求调整
    d2f_dx2 = (f(x + h, y) - 2 * f(x, y) + f(x - h, y)) / (h ** 2) # 对x求二阶偏导
    d2f_dy2 = (f(x, y + h) - 2 * f(x, y) + f(x, y - h)) / (h ** 2) # 对y求二阶偏导
    d2f_dxdy = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ** 2) # 对xy求混合偏导
    # 根据公式计算高斯曲率K和平均曲率H
    df_dx, df_dy = gradient(f, x, y) # 调用梯度函数求一阶偏导
    E = 1 + df_dx ** 2
    F = df_dx * df_dy
    G = 1 + df_dy ** 2
    L = d2f_dx2
    M = d2f_dxdy
    N = d2f_dy2
    K = (L * N - M ** 2) / (E * G - F ** 2) # 高斯曲率
    H = (E * N + G * L - 2 * F * M) / (2 * (E * G - F ** 2)) # 平均曲率
    return K, H

# 测试曲率函数
x = 0.5 # 某一点的x坐标
y = 0.5 # 某一点的y坐标
K, H = curvature(f, x, y) # 计算该点的曲率
print(f"高斯曲率为：{K:.6f}")
print(f"平均曲率为：{H:.6f}")
'''


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def surface_curvature(X, Y, Z):

	(lr, lb) = X.shape

	print(lr)
	print(lb)
# 一阶导数
	Xv, Xu = np.gradient(X)
	Yv, Yu = np.gradient(Y)
	Zv, Zu = np.gradient(Z)
# print(Xu)

# 二阶导数
	Xuv, Xuu = np.gradient(Xu)
	Yuv, Yuu = np.gradient(Yu)
	Zuv, Zuu = np.gradient(Zu)

	Xvv, Xuv = np.gradient(Xv)
	Yvv, Yuv = np.gradient(Yv)
	Zvv, Zuv = np.gradient(Zv)

# 2D 到 1D 转换
# 重构为一维向量
	Xu = np.reshape(Xu, lr*lb)
	Yu = np.reshape(Yu, lr*lb)
	Zu = np.reshape(Zu, lr*lb)
	Xv = np.reshape(Xv, lr*lb)
	Yv = np.reshape(Yv, lr*lb)
	Zv = np.reshape(Zv, lr*lb)
	Xuu = np.reshape(Xuu, lr*lb)
	Yuu = np.reshape(Yuu, lr*lb)
	Zuu = np.reshape(Zuu, lr*lb)
	Xuv = np.reshape(Xuv, lr*lb)
	Yuv = np.reshape(Yuv, lr*lb)
	Zuv = np.reshape(Zuv, lr*lb)
	Xvv = np.reshape(Xvv, lr*lb)
	Yvv = np.reshape(Yvv, lr*lb)
	Zvv = np.reshape(Zvv, lr*lb)

	Xu = np.c_[Xu, Yu, Zu]
	Xv = np.c_[Xv, Yv, Zv]
	Xuu = np.c_[Xuu, Yuu, Zuu]
	Xuv = np.c_[Xuv, Yuv, Zuv]
	Xvv = np.c_[Xvv, Yvv, Zvv]

# 曲面的第一基本系数(E,F,G)
	
	E = np.einsum('ij,ij->i', Xu, Xu)
	F = np.einsum('ij,ij->i', Xu, Xv)
	G = np.einsum('ij,ij->i', Xv, Xv)

	m = np.cross(Xu, Xv, axisa=1, axisb=1)
	p = np.sqrt(np.einsum('ij,ij->i', m, m))
	n = m/np.c_[p, p, p]
# n 为法向量
# 曲面的第二基本系数(L,M,N)， (e,f,g)
	L = np.einsum('ij,ij->i', Xuu, n)  # e
	M = np.einsum('ij,ij->i', Xuv, n)  # f
	N = np.einsum('ij,ij->i', Xvv, n)  # g

# 高斯曲率
	K = (L*N-M**2)/(E*G-F**2)
	K = np.reshape(K, lr*lb)
# print(K.size)

# 平均曲率
	H = ((E*N + G*L - 2*F*M)/((E*G - F**2)))/2
	print(H.shape)
	H = np.reshape(H,lr*lb)
# print(H.size)

# 主曲率
	Pmax = H + np.sqrt(H**2 - K)
	Pmin = H - np.sqrt(H**2 - K)
# [Pmax, Pmin]最大主曲率和最小主曲率
	Principle = [Pmax, Pmin]
	return Principle


x = np.linspace(-1, 1, 20)
y = np.linspace(-1, 1, 20)
[x, y] = np.meshgrid(x, y)

z = (x**3 +y**2 +x*y)

temp1 = surface_curvature(x, y, z)
#print("maximum curvatures")
#print(temp1[0])
#print("minimum curvatures")
#print(temp1[1])

print("x",x)
print("y",y)
print("z",z)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})


ax.plot_surface(x, y, z)
plt.show()

