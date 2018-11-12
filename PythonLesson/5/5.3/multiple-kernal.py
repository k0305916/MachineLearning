from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

# 采用RBF核函数实现的PCA进行降维时存在一个问题，就是我们必须指定先验参数r需要通过实验来找到一个合适的r值，
# 最好是通过参数调优算法来确定，例如网格搜索法，


def rbf_kernel_pca(x, gamma, n_components):
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
    
    lambdas:list
        Bigenvalues
    """
    # Calculate pairwise squared Euclidean distance
    # in the M*N dimensional dataset.
    sq_dists = pdist(x, 'sqeuclidean')

    # Convert pairwise distances into a square matrix.
    mat_sq_dists = squareform(sq_dists)

    # Compute the symmetric kernel matrix.
    K = exp(-gamma * mat_sq_dists)

    # Center the kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N, N))/N
    K = K-one_n.dot(K)-K.dot(one_n)+one_n.dot(K).dot(one_n)

    # Obtaining eigenpairs from the centered kernel matrix
    # numpy.eigh returns them in sorted order
    eigvals, eigvecs = eigh(K)

    # Collect the top k eigenvectors (projected samples)
    alphas = np.column_stack((eigvecs[:, -i]
                              for i in range(1, n_components+1)))

    # Collect the corresponding eigenvalues
    lambdas = [eigvals[-i] for i in range(1, n_components+1)]

    return alphas, lambdas


def project_x(x_new, x, gamma, alphas, lambdas):
    pair_dist = np.array([np.sum((x_new-row)**2) for row in x])
    k = np.exp(-gamma * pair_dist)
    return k.dot(alphas/lambdas)


from sklearn.datasets import make_moons
from matplotlib.ticker import FormatStrFormatter

x, y = make_moons(n_samples=100, random_state=123)
alphas, lambdas = rbf_kernel_pca(x, gamma=15, n_components=1)
x_new = x[25]
x_proj = alphas[25]  # original projection

x_reproj = project_x(x_new, x, gamma=15, alphas=alphas, lambdas=lambdas)
# plt.scatter(alphas[y == 0, 0], np.zeros((50)),
#             color='red', marker='^', alpha=0.5)
# plt.scatter(alphas[y == 1, 0], np.zeros((50)),
#             color='blue', marker='o', alpha=0.5)
# plt.scatter(x_proj, 0, color='black', marker='^', s=100,
#             label='original projection of point x[25]')
# plt.scatter(x_reproj, 0, color='green', marker='x',
#             s=500, label='remapped point x[25]')
# plt.legend(scatterpoints=1)
# plt.show()


# scikit-learn.decomposition
from sklearn.decomposition import KernelPCA
x,y = make_moons(n_samples=100,random_state=123)
scikit_kpca = KernelPCA(n_components=2,kernel='rbf',gamma=15)
x_skernpca = scikit_kpca.fit_transform(x)
plt.scatter(x_skernpca[y == 0, 0], x_skernpca[y==0,1],
            color='red', marker='^', alpha=0.5)
plt.scatter(x_skernpca[y == 1, 0], x_skernpca[y==1,1],
            color='blue', marker='o', alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

# scikit-learn实现了一些高级的非线性降维技术，
# 这些内容已经超出了本书的范围。
# 读者可以通过链接 http://scikit-learn.org/stable/modules/manifold.html
# 来了解相关内容概述及其示例。
