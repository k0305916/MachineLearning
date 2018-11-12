# 利用核PCA，我们可以通过非线性映射将数据转换到一个高维空间，
# 然后在此高维空间中使用标准PCA将其映射到另外一个低维空间中，
# 并通过线性分类器对样本进行划分（前提条件是，样本可根据输入空间的密度进行划分）。
# 这种方法的缺点是会带来高昂的计算成本，这也正是我们为什么要使用核技巧的原因。


from scipy.spatial.distance import pdist,squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

# 采用RBF核函数实现的PCA进行降维时存在一个问题，就是我们必须指定先验参数r需要通过实验来找到一个合适的r值，
# 最好是通过参数调优算法来确定，例如网格搜索法，
def rbf_kernel_pca(x,gamma,n_components):
    """
    RBF kernel PCA implementation.

    Parameters
    ----------
    x: {Numpy ndarray}, shape={n_samples,n_features}

    gamma: float
        Tuning parameter of the RBF kernel

    n_components: int
        Number of principal components to return

    Returns
    ---------
    x_pc: {Numpy ndarray},shape={n_samples,k_features}
        Projected dataset
    
    """
    # Calculate pairwise squared Euclidean distance
    # in the M*N dimensional dataset.
    sq_dists = pdist(x,'sqeuclidean')

    # Convert pairwise distances into a square matrix.
    mat_sq_dists = squareform(sq_dists)

    # Compute the symmetric kernel matrix.
    K = exp(-gamma * mat_sq_dists)

    # Center the kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N,N))/N
    K = K-one_n.dot(K)-K.dot(one_n)+one_n.dot(K).dot(one_n)

    # Obtaining eigenpairs from the centered kernel matrix
    # numpy.eigh returns them in sorted order
    eigvals,eigvecs = eigh(K)

    # Collect the top k eigenvectors (projected samples)
    x_pc = np.column_stack((eigvecs[:,-i] for i in range(1,n_components+1)))

    return x_pc


# sample 1: 分离半月形数据
from sklearn.datasets import make_moons
x,y = make_moons(n_samples=100,random_state=123)
# plt.scatter(x[y==0,0],x[y==0,1],color='red',marker='^',alpha=0.5)
# plt.scatter(x[y==1,0],x[y==1,1],color='blue',marker='o',alpha=0.5)
# plt.show()


from sklearn.decomposition import PCA
scikit_pca = PCA(n_components=2)
x_spca = scikit_pca.fit_transform(x)
# fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(7,3))
# ax[0].scatter(x_spca[y==0,0],x_spca[y==0,1],color='red',marker='^',alpha=0.5)
# ax[0].scatter(x_spca[y==1,0],x_spca[y==1,1],color='blue',marker='o',alpha=0.5)
# ax[1].scatter(x_spca[y==0,0],np.zeros((50,1))+0.02,color='red',marker='^',alpha=0.5)
# ax[1].scatter(x_spca[y==1,0],np.zeros((50,1))-0.02,color='blue',marker='o',alpha=0.5)
# ax[0].set_xlabel('PC1')
# ax[0].set_ylabel('PC2')
# ax[1].set_ylim([-1,1])
# ax[1].set_yticks([])
# ax[1].set_xlabel('PC1')
# plt.show()


# rbf_kernel_pca
from matplotlib.ticker import FormatStrFormatter
x_kpca = rbf_kernel_pca(x,gamma=15,n_components=2)
# fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(7,3))
# ax[0].scatter(x_kpca[y==0,0],x_kpca[y==0,1],color='red',marker='^',alpha=0.5)
# ax[0].scatter(x_kpca[y==1,0],x_kpca[y==1,1],color='blue',marker='^',alpha=0.5)
# ax[1].scatter(x_kpca[y==0,0],np.zeros((50,1))+0.02,color='red',marker='^',alpha=0.5)
# ax[1].scatter(x_kpca[y==1,0],np.zeros((50,1))-0.02,color='blue',marker='o',alpha=0.5)
# ax[0].set_xlabel('PC1')
# ax[0].set_ylabel('PC2')
# ax[1].set_ylim([-1,1])
# ax[1].set_yticks([])
# ax[1].set_xlabel('PC1')
# ax[0].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
# ax[1].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
# plt.show()


# sample 2: 分离同心圆
from sklearn.datasets import make_circles
x,y = make_circles(n_samples=1000,random_state=123,noise=0.1,factor=0.2)
plt.scatter(x[y==0,0],x[y==0,1],color='red',marker='^',alpha=0.5)
plt.scatter(x[y==1,0],x[y==1,1],color='blue',marker='o',alpha=0.5)
plt.show()


scikit_pca = PCA(n_components=2)
x_spca = scikit_pca.fit_transform(x)
fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(7,3))
ax[0].scatter(x_spca[y==0,0],x_spca[y==0,1],color='red',marker='^',alpha=0.5)
ax[0].scatter(x_spca[y==1,0],x_spca[y==1,1],color='blue',marker='o',alpha=0.5)
ax[1].scatter(x_spca[y==0,0],np.zeros((500,1))+0.02,color='red',marker='^',alpha=0.5)
ax[1].scatter(x_spca[y==1,0],np.zeros((500,1))-0.02,color='blue',marker='o',alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1,1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
plt.show()


x_kpca = rbf_kernel_pca(x,gamma=15,n_components=2)
fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(7,3))
ax[0].scatter(x_kpca[y==0,0],x_kpca[y==0,1],color='red',marker='^',alpha=0.5)
ax[0].scatter(x_kpca[y==1,0],x_kpca[y==1,1],color='blue',marker='^',alpha=0.5)
ax[1].scatter(x_kpca[y==0,0],np.zeros((500,1))+0.02,color='red',marker='^',alpha=0.5)
ax[1].scatter(x_kpca[y==1,0],np.zeros((500,1))-0.02,color='blue',marker='o',alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1,1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
ax[0].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
ax[1].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
plt.show()