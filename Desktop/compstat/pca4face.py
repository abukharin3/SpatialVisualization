import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA

class pca4face:
    def __init__(self, data, name):
        '''
        Class to perform PCA on face data set for face recognition.
        Includes downsampling on images
        '''
        self.name = name
        self.data = self.downsampling(data) # [num_samples, dim1, dim2]
        self.mean = self.get_mean()
        self.pca = self.get_pca()

    def get_mean(self):
        total = np.sum(self.data, axis = 0)
        mean = total / np.shape(self.data)[0]
        plt.imshow(mean)
        plt.savefig(self.name + "_mean.png")
        return mean

    def downsampling(self, data):
        # Downsampling from 243 * 320 to 60 * 80
        new_data = []
        for i in range(len(data)):
            trial = data[0]
            new = trial[::4, ::4]
            new_data.append(new)
        return np.array(new_data)

    def get_eigenfaces(self):
        data = self.data.reshape(len(self.data), 61 * 80)
        X = np.matmul(data.T, data)
        pca = PCA(n_components = 6)
        pca.fit(X)
        new = pca.transform(X)
        print(np.shape(new))
        ny = new.reshape(61, 80, 6)
        for i in range(6):
            plt.imshow(ny[:, :, i])
            plt.savefig("eigenface" + str(i) + ".png")


    def get_pca(self):
        data = self.data.reshape(len(self.data), 61 * 80)
        X = np.matmul(data.T, data)
        pca = PCA(n_components = 1)
        pca.fit(X)
        return pca

    def classify(self, other_image):
        trial = self.pca.transform(other_image)
        return (abs(trial[0][0]))



data1 = []
data2 = []
path = r"C:\Users\Alexander\Desktop\compstat\yalefaces"
for file_name in os.listdir(path):
    if "subject14" in file_name and "test" not in file_name:
        im = Image.open(path + "\\" + file_name)
        im_array = np.array(im)
        data1.append(im_array)
    elif "subject01" in file_name:
        im = Image.open(path + "\\" + file_name)
        im_array = np.array(im)
        data2.append(im_array)
    elif "test" in file_name:
        test = Image.open(path + "\\" + file_name)
        test_array = np.array(test)
data1 = np.array(data1)
data2 = np.array(data2)

test_array = test_array[::4, ::4].reshape(1, 4880)
pc1 = pca4face(data1, "pc1")
pc2 = pca4face(data2, "pc2")
pc1.get_eigenfaces()
print("Subject14:", pc1.classify(test_array))
print("Subject01:", pc2.classify(test_array))



