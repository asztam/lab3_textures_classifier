# textures classifier, lab3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.decomposition import PCA
from glob import glob
from skimage.feature import graycomatrix, graycoprops
from os.path import sep, splitext, join
from itertools import product
from PIL import Image
from pandas import DataFrame


features_names = ['dissimilarity', 'correlation', 'contrast', 'energy', 'homogeneity', 'ASM']
distances = (1, 3, 5)    # pixel distances
angles = (0, np.pi/4, np.pi/2, 3*np.pi/4)   # four directions -> 0, 45, 90, 135 degrees


def glcm_array(patch):
    patch_64 = (patch / np.max(patch) * 63).astype('uint8')
    comatrix = graycomatrix(patch_64, distances, angles, 64, True, True)
    features_vector_: list = []
    for features_ in features_names:
        features_vector_.extend(list(graycoprops(comatrix, features_).flatten()))
    return features_vector_


def full_name():
    distances_string = ('1', '3', '5')
    angles_string = ['0 deg', '45 deg', '90 deg', '135 deg']
    return['_'.join(n) for n in product(features_names, distances_string, angles_string)]

# ============= Textures sampling =============


textures_folder = "textures"


samples_folder = "samples"


path = glob(textures_folder + "\\*\\*.jpg")
split = [s.split(sep) for s in path]
_, categories, files = zip(*split)
size = 128, 128
no_of_samples = 20

features = []
for category, infile in zip(categories, files):
    image = Image.open(join(textures_folder, category, infile))
    xs = np.random.randint(0, image.width - size[0], no_of_samples)
    ys = np.random.randint(0, image.height - size[1], no_of_samples)
    name, _ = splitext(infile)
    for i, (x, y) in enumerate(zip(xs, ys)):
        sample_image = image.crop((x, y, x + size[0], y + size[1]))
        sample_image.save(join(samples_folder, category, f'{name:s}_{i:02d}.jpg'))
        image_gray = np.array(image.convert('L'))
        features_vector = glcm_array(image_gray)
        features_vector.append(category)
        features.append(features_vector)


features_names = full_name()
features_names.append('Category')

data_frame = DataFrame(data=features, columns=features_names)
data_frame.to_csv('textures_data.csv', sep=',', index=False)

features = pd.read_csv('textures_data.csv', sep=',')

# ============= Visualization =============

data = np.array(features)
x = (data[:, :-1]).astype('float64')
y = data[:, -1]

x_transform = PCA(n_components=3)

xt = x_transform.fit_transform(x)

red = y == 'sonoma'
blue = y == 'tkanina'
cyan = y == 'tynk'

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xt[red, 0], xt[red, 1], xt[red, 2], c="r")
ax.scatter(xt[blue, 0], xt[blue, 1], xt[blue, 2], c="b")
ax.scatter(xt[cyan, 0], xt[cyan, 1], xt[cyan, 2], c="c")


# ============= Classification =============

classifier = svm.SVC(gamma='auto')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

cm = confusion_matrix(y_test, y_pred, normalize='true')
print(cm)

disp = plot_confusion_matrix(classifier, x_test, y_test, cmap=plt.cm.Blues)
plt.show()
