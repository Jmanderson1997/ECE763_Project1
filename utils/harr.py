import numpy as np
import matplotlib.pyplot as plt
import pickle
from os import path
from skimage.transform import integral_image
from skimage.feature import haar_like_feature, haar_like_feature_coord, draw_haar_like_feature
from sklearn.ensemble import RandomForestClassifier
from dataset.data import get_pickled_data
from sklearn.metrics import f1_score
from utils.pathing import get_p2_pickle_folder
from progressbar import progressbar

ALL_FEATURE_TYPES = ['type-2-x', 'type-2-y', 'type-3-x', 'type-3-y', 'type-4']

def plot_harr_feat(image, coords):
    for coord in coords:
        fig, ax = plt.subplots(1,1)
        im = draw_haar_like_feature(image, 0, 0, 20, 20, [coord])
        ax.imshow(im)
        plt.show()
    return


def extract_harr_features(image, feature_types=None, feature_coords=None):
    ii = integral_image(image)
    return haar_like_feature(ii, 0,0, 20, 20, feature_type=feature_types, feature_coord=feature_coords)


def pickle_harr_features(feature_types=None, feature_coords=None):
    train_faces, train_background, test_faces, test_background = get_pickled_data(flatten=False)
    face_features = []
    back_features = []
    print("Pickling Face Harr Features")
    for image in progressbar(np.concatenate((train_faces, test_faces))):
        face_features.append(extract_harr_features(image, feature_types, feature_coords))
    print("Pickling Background Harr Features")
    for image in progressbar(np.concatenate((train_background, test_background))):
        back_features.append(extract_harr_features(image))
    np.save(path.join(get_p2_pickle_folder(),'face_harr_features'), np.array(face_features))
    np.save(path.join(get_p2_pickle_folder(),'back_harr_features'), np.array(back_features))


def load_harr_features(split=None):
    if not path.exists(path.join(get_p2_pickle_folder(),'face_harr_features.npy')):
        pickle_harr_features()
    face = np.load(path.join(get_p2_pickle_folder(),'face_harr_features.npy'))
    back = np.load(path.join(get_p2_pickle_folder(),'back_harr_features.npy'))
    if split is None:
        return face, back, None, None
    else:
        return face[:int(len(face)*split)], back[:int(len(face)*split)], face[int(len(face)*split):], back[int(len(face)*split):]


def get_best_threshold(feature_samples, labels):
    best_acc = 0
    best_threshold = None
    for threshold in np.unique(feature_samples):
        decisions = feature_samples > threshold
        acc = np.mean(decisions == labels)
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold

    return best_threshold, best_acc


def find_key_features(plot=True, n_features=20, pickle_feats=False):
    print("Finding key Harr Features")
    train_faces, train_background, test_faces, test_background = load_harr_features(split=.9)
    data = np.concatenate((train_faces, test_faces, train_background, test_background))
    labels = np.concatenate((np.ones(len(train_faces)+len(test_faces)),
                             np.zeros(len(train_background)+len(test_background))))

    # feat_indx = np.arange(0, len(train_faces[0]), 1)
    # accuracies = []
    # thresholds = []
    # for idx in progressbar(feat_indx):
    #     acc, threshold = get_best_threshold(data[:,idx], labels)
    #     accuracies.append(acc)
    #     thresholds.append(threshold)
    #
    # best_features = np.flip(np.argsort(accuracies))

    print("Fitting Tree Model")
    clf = RandomForestClassifier(n_estimators=1000, max_depth=None, n_jobs=-1, random_state=0)
    clf.fit(data, labels)
    best_features = np.flip(np.argsort(clf.feature_importances_))

    if pickle_feats:
        np.save(path.join(get_p2_pickle_folder(), 'key_harr_feat_idx_rf_2'), best_features)
        feat_coord, feat_type = haar_like_feature_coord(20, 20)
        sel_feat_coords = feat_coord[best_features]
        sel_feat_types = feat_type[best_features]
        np.save(path.join(get_p2_pickle_folder(), 'key_harr_feat_coords'), sel_feat_coords)
        with open(path.join(get_p2_pickle_folder(), 'key_harr_feat_types'), 'wb') as f:
            pickle.dump(sel_feat_types, f)

    if plot:
        train_faces, train_background, test_faces, test_background = get_pickled_data(flatten=False)
        feat_coord, feat_type = haar_like_feature_coord(20, 20)
        sel_feat_coords = feat_coord[best_features]

        plot_harr_feat(train_faces[0], sel_feat_coords)

    return best_features


def get_pickled_key_features(just_idx=True):
    if just_idx:
        if not path.exists(path.join(get_p2_pickle_folder(), 'key_harr_feat_idx_rf_2.npy')):
            best_feat = find_key_features(pickle_feats=True)
            return best_feat
        return np.load(path.join(get_p2_pickle_folder(), 'key_harr_feat_idx_rf_2.npy'))
    else:
        return


if __name__ == '__main__':
    pickle_harr_features()
    # find_key_features(pickle_feats=True)