import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.gaussian_process.kernels import WhiteKernel

TEST_CASE_NUMBER = 100
FOLD = 5
INPUT_DIM = 2
SIGNAL_LEVEL = 50
NOISE_LEVEL = 1


def _init():
    """
        Gaussain Process에 필요한 것들을 initialize합니다.
    """
    # 이 코드가 있으면, 랜덤 값이 특정 값으로 고정됩니다.
    # np.random.seed(1)

    # Observations할 X 값을 설정합니다.
    X = np.random.rand(TEST_CASE_NUMBER * (FOLD - 1), INPUT_DIM) * SIGNAL_LEVEL

    # 관측
    y = f(X).ravel()
    y += np.random.normal(0, NOISE_LEVEL, size=y.shape)
    return X, y


def f(x):
    """
        예측해야 하는 함수입니다.
    """
    return np.matmul(x * np.absolute(np.sin(x)), np.array([[2], [1]]))


def _print_test_error(x, y_pred, sigma):
    print('confidence\terror')
    for z_value, confidence in [(0.5, 38), (1, 68), (1.96, 95)]:
        delta = np.maximum(
            np.absolute(y_pred - f(x).flatten()) - z_value * sigma,
            np.zeros(sigma.shape))
        error = np.count_nonzero(delta) * 100 / delta.shape[0]
        print('%9d%%\t%4d%%' % (confidence, error))


def main():
    X, y = _init()
    kernel = WhiteKernel(noise_level=1e-10) \
           + ConstantKernel(2.0, (1e-3, 1e3)) \
           * RBF([.1, .1], [(1e-5, 1e2), (1e-5, 1e2)])

    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gpr.fit(X, y)

    x = np.random.rand(TEST_CASE_NUMBER, INPUT_DIM) * SIGNAL_LEVEL
    y_pred, sigma = gpr.predict(x, return_std=True)
    _print_test_error(x, y_pred, sigma)
    print(np.average(y))
    print(np.average(sigma))


if __name__ == '__main__':
    main()
