import os
import numpy as np
from numpy.random import randint
from utils.pathing import *
from PIL import Image


def parse_ellipse_files(file, n_samples, grayscale=False):
    proj_dir = get_data_dir()
    file = os.path.join(get_fold_dir(), file)
    faces = []
    background= []

    with open(file, 'r') as f:
        while True:
            im_path = f.readline()[:-1]+".jpg"
            if im_path == '.jpg':
                break
            else:
                im_path = path.join(proj_dir, im_path)
            n_imgs = int(f.readline()[:-1])
            im = Image.open(im_path)
            if grayscale:
                im = im.convert('L')
            # im.save(path.join(get_pickle_folder(), 'original'+'.png'), format='png')
            for i in range(n_imgs):
                if im.mode != 'RGB' and not grayscale:
                    skip = f.readline()
                    continue
                height, width, _, center_x, center_y, _ = list(map(float, f.readline().split()))
                box = (int(max(center_x-width, 0)), int(max(center_y-height, 0)), int(min(center_x+width, im.width)), int(min(center_y+height, im.height)))
                face = im.crop(box)
                # face.save(path.join(get_pickle_folder(), 'pic'+str(i)+'.png'), format='png')
                face = face.resize((20,20))
                # face.save(path.join(get_pickle_folder(), 'resized' + str(i)+'.png'), format='png')
                faces.append(np.array(face))

                r_left = randint(0, im.width-20)
                r_top = randint(0, im.height-20)
                r_right = randint(r_left+20, im.width)
                r_bottom = randint(r_top+20, im.height)
                no_face = im.crop((r_left, r_top, r_right, r_bottom))
                # no_face.save(path.join(get_pickle_folder(), 'back'+str(i)+'.png'), format='png')
                no_face = no_face.resize((20,20))
                # no_face.save(path.join(get_pickle_folder(), 'b_resized' + str(i)+'.png'), format='png')
                no_face = np.array(no_face)
                background.append(no_face)


                n_samples -= 2
                if n_samples <=0:
                    return faces, background

    return faces, background


def pickle_face_data(n_samples=30000, grayscale=False):
    faces = []
    background = []
    ellipse_files = [file for file in os.listdir(get_fold_dir()) if file.endswith("ellipseList.txt")]
    file_index = 0
    # parse_ellipse_files(ellipse_files[6], 1)
    while len(faces)+len(background) < n_samples:
        pos, neg = parse_ellipse_files(ellipse_files[file_index], n_samples - (len(background)+len(faces)), grayscale=grayscale)
        faces += pos
        background += neg
        file_index += 1

    dataset_folder = get_dataset_dir()
    np.save(os.path.join(dataset_folder, 'faces'), faces)
    np.save(os.path.join(dataset_folder, 'background'), background)


def get_pickled_data(flatten=True):
    faces = np.load(os.path.join(get_dataset_dir(), 'faces.npy'))
    background = np.load(os.path.join(get_dataset_dir(), 'background.npy'))
    if flatten:
        faces = faces.reshape(-1, 20*20*3)
        background = background.reshape(-1, 20*20*3)
    return faces[:int(len(faces)*.9)], background[:int(len(background)*.9)], faces[int(len(faces)*.9):], background[int(len(background)*.9):]


if __name__ == '__main__': 
    pickle_face_data(200, grayscale=True)
