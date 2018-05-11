import os
import nibabel as nib
import numpy as np
from skimage.transform import resize
import json
import pdb
from glob import glob
with open('info.json') as f:
    ParSet = json.load(f)

def Normarlize(xs):

	xs_min = np.min(xs)
	xs_max = np.max(xs)
	xs = (xs - xs_min)/(xs_max - xs_min)
	return xs

def data_load_train(base_path=ParSet['TRAIN_PATH'], nii_index=0):

	label_path = [base_path + p for p in os.listdir(base_path) if p.endswith('Label.nii.gz')]
	image_path = [p.replace('Label', '') for p in label_path]
	prob_path = [p.replace('Label', '_prob') for p in label_path]

	xs, ys, img_prob = [nib.load(p[nii_index]).get_data() for p in [image_path, label_path, prob_path]]
	pdb.set_trace()
	xs = Normarlize(xs)

	xs_shape = np.shape(xs)

	label_aux1_shape = [xs_shape[0]//2, xs_shape[1]//2, xs_shape[2]//2]

	label_aux2_shape = [xs_shape[0]//4, xs_shape[1]//4, xs_shape[2]//4]

	label_aux3_shape = [xs_shape[0]//8, xs_shape[1]//8, xs_shape[2]//8]

	label_aux1 = resize(ys, label_aux1_shape, mode='reflect')

	label_aux2 = resize(ys, label_aux2_shape, mode='reflect')

	label_aux3 = resize(ys, label_aux3_shape, mode='reflect')

	xs, ys, img_prob, label_aux1, label_aux2, label_aux3 = [item[np.newaxis, ..., np.newaxis] for item in [xs, ys, img_prob, label_aux1, label_aux2, label_aux3]]

	xs_temp = np.zeros((ParSet['BATCH_SIZE'], xs_shape[0], xs_shape[1], xs_shape[2], ParSet['CHANNEL_NUM']))
	pdb.set_trace()
	xs_temp[:, :, :, :, 0] = xs[:, :, :, :, 0]

	xs_temp[:, :, :, :, 1] = img_prob[:, :, :, :, 0]

	return xs_temp, ys, label_aux1, label_aux2, label_aux3


def load_inference(base_path=ParSet['TEST_PATH'], nii_index=0):

	pair_list = glob('{}/*_prob.nii.gz'.format(base_path))
	pair_list.sort()
	prob_name = pair_list[nii_index]

	image_list = [p.replace('_prob', '') for p in pair_list]
	image_name = prob_name[nii_index]


	xs = nib.load(image_name).get_data()
	prob_image = nib.load(prob_name).get_data()

	xs_shape = np.shape(xs)

	xs = Normarlize(xs)

	seq = image_name.split("/")[-1].split(".")[0]

	num_dex = int(seq)

	xs, prob_image = [item[np.newaxis, ..., np.newaxis] for item in [xs, prob_image]]

	xs_temp = np.zeros((ParSet['BATCH_SIZE'], xs_shape[0], xs_shape[1], xs_shape[2], ParSet['CHANNEL_NUM']))

	xs_temp[:, :, :, :, 0] = xs[..., 0]

	xs_temp[:, :, :, :, 1] = prob_image[..., 0]

	return xs_temp,num_dex