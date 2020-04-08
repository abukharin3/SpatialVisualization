import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


class EM():
    '''
    EM Algorithm for MNIST Dataset, each image is of length 784. We distinguish between 2s and 6s
    '''
    def __init__(self):
        # Initialize variable
        self.mu2 = np.random.normal(0, 1, 784)
        self.mu6 = np.random.normal(0, 1, 784)
        self.cov2 = np.identity(784)
        self.cov6 = np.identity(784)
        self.pi2 = 0.5
        self.pi6 = 0.5
        self.probs = np.array([1190, 2])

    def multivariate_normal(self, image):
        # We use a low rank approximation
        diagonal2 = np.diag(self.cov2).copy()
        diagonal6 = np.diag(self.cov6).copy()
        # Diagonal may contain zeroes, so we take those out
        for i in range(len(diagonal2)):
            if diagonal2[i] <= 0:
                diagonal2[i] = 1.0
            if diagonal6[i] <= 0:
                diagonal6[i] = 1.0

        # Calculate determinants for each covariance matrix
        det2 = np.prod(diagonal2)
        det6 = np.prod(diagonal6)

        # Calculate low-rank approximation of inverses
        inv2 = np.diag(1.0 / diagonal2)
        inv6 = np.diag(1.0 / diagonal6)

        # Calculate constant part of normal
        constant6 = 1.0 / np.sqrt(det6)
        constant2 = 1.0 / np.sqrt(det2)

        # Calculate exponent part of normal
        exponent2 = np.exp(-0.5 * np.matmul(np.matmul((image - self.mu2).transpose(), inv2), (image - self.mu2)))
        exponent6 = np.exp(-0.5 * np.matmul(np.matmul((image - self.mu6).transpose(), inv6), (image - self.mu6)))

        return np.array([1 / constant2 * exponent2, 1 / constant6 * exponent6])


    def calculate_probs(self, data):
        #Data (num images, num classes)
        probs = []
        for i in range(len(data[0])):
            prob = self.multivariate_normal(data[:, i]) / np.sum(self.multivariate_normal(data[:, i]))
            probs.append(prob / np.linalg.norm(prob))
        self.probs = np.array(probs)


    def update_mean(self, data):
        # Calculate mean
        mu2 = np.sum(self.probs[:, 0] * data, axis = 1) / np.sum(self.probs[:, 0])
        mu6 = np.sum(self.probs[:, 1] * data, axis = 1) / np.sum(self.probs[:, 1])
        self.mu2 = mu2
        self.mu6 = mu6

    def update_variance(self, data):
        # Calculate covariance
        cov2 = np.zeros([784,784])
        cov6 = np.zeros([784,784])
        reg2 = np.sum(self.probs[:, 0])
        reg6 = np.sum(self.probs[:, 1])
        for j in range(1190):
            cov2 = cov2 + self.probs[j, 0] * np.outer((self.mu2 - data[:, j]), (self.mu2 - data[:, j])) / reg2
            cov6 = cov6 + self.probs[j, 1] * np.outer((self.mu6 - data[:, j]), (self.mu6 - data[:, j])) / reg6

        self.cov2 = cov2
        self.cov6 = cov6
        #print(self.cov2)

    def update_pis(self, data):
        # Calculate pis
        self.pi2 = np.sum(self.probs[:, 0]) / len(data[0])
        self.pi6 = np.sum(self.probs[:, 1]) / len(data[0])

    def loss(self, data):
        # Calculate loss
        loss2 = 0
        loss6 = 0
        for j in range(1990):
            loss2 += self.probs[:, 0][j] * self.pi2 + self.probs[:, 0][j] * self.multivariate_normal(data[:, 0])[0]
            loss6 += self.probs[:, 1][j] * self.pi2 + self.probs[:, 1][j] * self.multivariate_normal(data[:, 0])[1]
        return loss2 + loss6

    def optimize(self, data):
        likelihood = []
        for i in range(30):
            self.calculate_probs(data)
            self.update_mean(data)
            #self.update_variance(data)
            self.update_pis(data)
            image1 = self.mu2.reshape(28,28)
            plt.imshow(image1.transpose())
            plt.show()
            image2 = self.mu6.reshape(28,28)
            plt.imshow(image2.transpose())
            plt.show()
            likelihood.append(self.loss(data))
        plt.plot(likelihood)
        plt.show()
        image1 = self.mu2.reshape(28,28)
        plt.imshow(image1.transpose())
        plt.show()
        image2 = self.mu6.reshape(28,28)
        plt.imshow(image2.transpose())
        plt.show()



def main():
    # Load data
    data = np.loadtxt("data.dat")
    labels = np.loadtxt("label.dat")
    # Visualize images
    number = data[:, 300]
    image = number.reshape(28,28)
    plt.imshow(image.transpose())
    plt.show()
    plt.savefig("two.png")
    a = EM()
    a.optimize(data)


if __name__== "__main__":
    main()
