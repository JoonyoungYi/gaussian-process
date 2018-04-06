import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.gaussian_process.kernels import WhiteKernel


def _init():
    """
        Gaussain Process에 필요한 것들을 initialize합니다.
    """
    # 이 코드가 있으면, 랜덤 값이 특정 값으로 고정됩니다.
    np.random.seed(1)

    # Observations할 X 값을 설정합니다.
    X = np.linspace(0.1, 9.9, 20)  # 0.1~9.9에서 등분으로 20개 뽑음.
    X = np.atleast_2d(X).T

    # 관측
    y = f(X).ravel()
    y += np.random.normal(0, 1, size=y.shape)
    return X, y


def f(x):
    """
        예측해야 하는 함수입니다.
    """
    return x * np.sin(x)


def main():
    X, y = _init()

    kernel = WhiteKernel(noise_level=1e-2)
    kernel += ConstantKernel(2.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))

    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gpr.fit(X, y)

    x = np.atleast_2d(np.linspace(0, 10, 1000)).T
    y_pred, sigma = gpr.predict(x, return_std=True)

    delta = np.maximum(
        np.absolute(y_pred - f(x).flatten()) - 1.96 * sigma,
        np.zeros(sigma.shape))
    print(np.count_nonzero(delta) * 100 / delta.shape[0])


if __name__ == '__main__':
    main()
