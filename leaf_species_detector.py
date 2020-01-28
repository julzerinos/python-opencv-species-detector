import os
import sys

import cv2 as cv
import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from skimage import feature
from sklearn.svm import SVC
from sklearn.metrics import classification_report

import mahotas as mt


plant_species = {
    0: 'circinatum',
    1: 'garryana',
    2: 'glabrum',
    3: 'kelloggii',
    4: 'macrophyllum',
    5: 'negundo'
}


def get_image_paths(parent_images_path='isolated'):
    leaves = []
    for spec in plant_species.values():
        plant_path = os.path.join(parent_images_path, spec)
        leaves.append([
            os.path.join(plant_path, p) for p in sorted(os.listdir(plant_path))
            ])
    return leaves


def fv_kaze(image_path, **kwargs):
    # Source https://medium.com/machine-learning-world/feature-extraction-and-similar-image-search-with-opencv-for-newbies-3c59796bf774
    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    alg = cv.KAZE_create()
    kps = alg.detect(image)

    kps = sorted(kps, key=lambda x: -x.response)[:kwargs['kaze_vector_size']]

    kps, dsc = alg.compute(image, kps)

    dsc = dsc.mean(axis=0)

    return dsc


def fv_histogram(image_path, **kwargs):
    # Source https://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/
    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    hist = cv.calcHist([image], [0], None, [256], [0, 256])
    hist = hist.astype("float")
    hist /= (hist.sum() + kwargs['hist_eps'])

    return hist.flatten()


def fv_local_binary_pattern(image_path, **kwargs):
    # Source https://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/
    
    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    lbp = feature.local_binary_pattern(
        image, kwargs['hist_numpoints'], kwargs['hist_radius'], method="uniform"
        )
    (hist, _) = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, kwargs['hist_numpoints'] + 3),
        range=(0, kwargs['hist_numpoints'] + 2)
        )

    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + kwargs['hist_eps'])

    # return the histogram of Local Binary Patterns
    return hist


def fv_haralick(image_path, **kwargs):
    # Source https://gogul.dev/software/texture-recognition
    image = cv.imread(image_path)

    textures = mt.features.haralick(image)
    return textures.mean(axis=0)


def fv_image_statistics(image_path, **kwargs):
    # Source https://github.com/AayushG159/Plant-Leaf-Identification/blob/master/Flavia%20py%20files/classify_leaves_flavia.ipynb  # noqa

    image = cv.imread(image_path)
    gs = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Preprocessing
    blur = cv.GaussianBlur(gs, (25, 25), 0)
    ret_otsu, im_bw_otsu = cv.threshold(
        blur, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    kernel = np.ones((50, 50), np.uint8)
    closing = cv.morphologyEx(im_bw_otsu, cv.MORPH_CLOSE, kernel)

    # Shape features
    contours, _ = cv.findContours(
        closing, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    _ = cv.moments(cnt)
    area = cv.contourArea(cnt)
    perimeter = cv.arcLength(cnt, True)
    x, y, w, h = cv.boundingRect(cnt)
    aspect_ratio = float(w)/h
    rectangularity = w*h/area
    circularity = ((perimeter)**2)/area

    # Color features
    red_channel = image[:, :, 2]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 0]
    blue_channel[blue_channel == 255] = 0
    green_channel[green_channel == 255] = 0
    red_channel[red_channel == 255] = 0

    red_mean = np.mean(red_channel)
    green_mean = np.mean(green_channel)
    blue_mean = np.mean(blue_channel)

    red_std = np.std(red_channel)
    green_std = np.std(green_channel)
    blue_std = np.std(blue_channel)

    #Texture features
    textures = mt.features.haralick(gs)
    ht_mean = textures.mean(axis=0)
    contrast = ht_mean[1]
    correlation = ht_mean[2]
    inverse_diff_moments = ht_mean[4]
    entropy = ht_mean[8]

    vector = [
        area,
        perimeter,
        w, h,
        aspect_ratio,
        rectangularity,
        circularity,
        red_mean,
        green_mean, blue_mean, red_std,
        green_std, blue_std,
        contrast, correlation,
        inverse_diff_moments, entropy
             ]

    return vector


def fv_hu_moments(image_path, **kwargs):
    # Source https://gogul.dev/software/image-classification-python
    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    feature = cv.HuMoments(cv.moments(image)).flatten()
    return feature


features_methods = {
    0: fv_kaze,
    1: fv_histogram,
    2: fv_local_binary_pattern,
    3: fv_haralick,
    4: fv_image_statistics,
    5: fv_hu_moments
}
kwargs = {
    'kaze_vector_size': 32,

    'hist_eps': 1e-7,
    'hist_numpoints': 24,
    'hist_radius': 8
}


def features_wrapper(fvs, l, **kwargs):
    fv_results = []
    for fv in fvs:
        fv_results.append(features_methods[fv](l, **kwargs))
    return np.hstack(fv_results)


if __name__ == '__main__':
    fvs = [int(i) for i in sys.argv[1].split(',')]

    leaves = get_image_paths()

    training = {
        'fv': [],
        'lab': []
    }

    for label in plant_species:
        print(f"Processing {plant_species[label]}")
        training['fv'].extend([
            features_wrapper(fvs, l, **kwargs) for l in leaves[label]
            ])
        training['lab'].extend([label] * len(leaves[label]))

    fv_train, fv_test, lab_train, lab_test = train_test_split(
        training['fv'], training['lab'],
        train_size=0.8, random_state=None
        )

    sc_X = StandardScaler()
    fv_train = sc_X.fit_transform(fv_train)
    fv_test = sc_X.transform(fv_test)

    model = SVC(
        C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape=None, degree=3, gamma='auto',
        kernel='rbf',
        max_iter=-1, probability=False, random_state=None,
        shrinking=True,
        tol=0.001, verbose=False)

    model.fit(fv_train, lab_train)
    score = model.score(fv_test, lab_test)

    print(f'Methods in model: {[features_methods[f].__name__ for f in fvs]}')
    print(f'Total accuracy of the model: {score}')
    print(classification_report(
        model.predict(fv_test),
        lab_test,
        target_names=plant_species.values())
        )
