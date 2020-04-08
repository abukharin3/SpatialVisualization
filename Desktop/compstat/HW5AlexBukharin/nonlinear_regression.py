import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

class NonlinearRegression:
    '''
    Class to preform nonlinear rgression with a spline function, using all
    of the data points
    '''
    def __init__(self, x, y, weights = 0, alpha = 0.5):
        self.x = list(self.sort(x, y)[0])
        self.y = list(self.sort(x, y)[1])
        self.weights = np.ones(len(x))
        self.h = self.inter_knot_distances(x)
        self.alpha = alpha
        self.M = self.get_M()
        self.Q = self.get_Q()
        self.sigma = self.get_sigma()

    def lin_reg(self):
        x = np.array(self.x)
        y = np.array(self.y)
        reg = linregress(x, y)
        slope = reg[0]
        intercept = reg[1]
        pred = []
        for coord in x:
            pred.append(intercept + slope * coord)
        pred = np.array(pred)
        return 1 / len(x) * np.sum((pred - y) ** 2)


    def sort(self, x, y):
        # For now naive bubble sort
        for i in range(len(x)):
            for j in range(i, len(x)):
                if x[i] > x[j]:
                    x[i], x[j] = x[j], x[i]
                    y[i], y[j] = y[j], y[i]
        return (x, y)

    def inter_knot_distances(self, x):
        h = []
        for i in range(len(x) - 1):
            h.append(x[i + 1] - x[i])
        return np.array(h)

    def get_M(self):
        sub_diag = np.array([self.h[i] / 6 for i in range(1 , len(self.h))])
        diag = np.array([1/3 * (self.h[i] - self.h[i + 1]) for i in range(len(self.h) - 1)])
        M = np.zeros([len(self.x) - 2, len(self.x) - 2])
        for i in range(len(diag)):
            if i == 0:
                M[i, i] = diag[i]
            else:
                M[i, i] = diag[i]
                M[i - 1, i] = sub_diag[i]
                M[i, i - 1] = sub_diag[i]
        return M

    def get_Q(self):
        first_diag = [1 / self.h[i] for i in range(len(self.h) - 1)]
        sec_diag = [-1 / self.h[i] - 1 / self.h[i + 1] for i in range(len(self.h) - 1)]
        third_diag = [1 / self.h[i] for i in range(1, len(self.h))]
        Q = np.zeros([len(self.x) - 2, len(self.x)])
        for i in range(len(self.x) - 2):
            Q[i, i] = first_diag[i]
            Q[i, i + 1] = sec_diag[i]
            Q[i, i + 2] = third_diag[i]
        return Q

    def get_sigma(self):
        W = np.diag(self.weights)
        # First we compute M inverse * Q
        first_step = np.matmul(np.linalg.inv(self.M), self.Q)
        # Now we calculate f_inverse
        f_inverse = self.alpha * W + (1 - self.alpha) * np.matmul(self.Q.T, first_step)
        # The invert it and multiply by W, y
        f = np.matmul(np.linalg.inv(f_inverse), self.alpha * np.matmul(W, self.y))
        sigma = list(np.matmul(first_step, f))
        sigma.insert(0, 0)
        sigma.append(0)
        return np.array(sigma)

    def binary_search(self, x_coord):
        # Method to find which interval point lies in
        upper = len(self.x)
        lower = 0
        mid = len(self.x) // 2
        while(upper - lower > 1):
            if self.x[mid] > x_coord:
                upper = mid
                mid = mid // 2
            else:
                lower = mid
                mid = lower + (upper - lower) // 2
        return (lower, upper)


    def predict(self, x_coord):
        # First we preform a binary search
        if x_coord in self.x:
            return self.y[self.x.index(x_coord)]
        else:
            interval = self.binary_search(x_coord)
            return self.spline(interval, x_coord)

    def spline(self, interval, x_coord):
        y1 = self.sigma[interval[0]] / (6 * self.h[interval[0]]) * (self.x[interval[1]] - x_coord) ** 3
        y2 = self.sigma[interval[1]] / (6 * self.h[interval[0]]) * (x_coord - self.x[interval[0]]) ** 3
        y3 = (self.y[interval[1]] / self.h[interval[0]] - self.sigma[interval[1]] * self.h[interval[0]] / 6) * (x_coord - self.x[interval[0]])
        y4 = (self.y[interval[0]] / self.h[interval[0]] - self.sigma[interval[0]] * self.h[interval[0]] / 6) * (self.x[interval[1]] - x_coord)
        y_hat =  y1 + y2 + y3 + y4
        return y_hat

    def printer(self):
        start = self.x[0]
        end = self.x[-1]
        length = end - start

        domain = [start + i / (1000) * length for i in range(1000)]
        interpolated = []
        for num in domain:
            interpolated.append(self.predict(num))
        plt.plot(domain, interpolated, 'x')
        plt.plot(self.x, self.y, 'o')
        plt.show()





data = np.loadtxt("copper-new.txt")
x = data.T[1]
y = data.T[0]
nl = NonlinearRegression(x, y, alpha = 0)
nl.printer()
print(nl.predict(400))

def leave_one_out_loss(alpha):
    loss = 0
    for index in range(len(x) - 1):
        new_x = np.concatenate((x[0: index], x[index + 1:]))
        new_y = np.concatenate((y[0: index], y[index + 1:]))
        new_spline = NonlinearRegression(new_x, new_y, alpha = alpha)
        y_hat = new_spline.predict(x[index])
        loss += (y[index] - y_hat) ** 2
    return loss

def cross_validate():
    min_loss = 10**400
    min_alpha = 0
    alphas = [i / 1000 for i in range(1000)]
    for alpha in alphas:
        loss = leave_one_out_loss(alpha)
        print(loss)
        if loss <= min_loss:
            min_alpha = alpha
            min_loss = loss
    return min_alpha


