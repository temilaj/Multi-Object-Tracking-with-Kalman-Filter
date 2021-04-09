import math
import numpy as np 
import cv2

bats_root_path = '../CS585-BatImages/Gray'

def generate_bat_image_list():
	image_list = []
	for i in range(750, 901):
		img_path = f"{bats_root_path}/CS585Bats-Gray_frame000000{i}.ppm"
		image_list.append(cv2.imread(img_path))
	print(f"{len(image_list)} bat image frames processed")
	return image_list

def fetch_bat_localizations(localizations_root_path):
	min_value = math.inf;
	localizations = []
	for i in range(750, 901):
		localization_path = f"{localizations_root_path}/CS585Bats-Localization_frame000000{i}.txt"
		loaded_file = np.loadtxt(localization_path, delimiter=',')
		min_value = min(len(loaded_file), min_value)
		localizations.append(loaded_file)
		# localizations[i] = np.loadtxt(file_list[i], delimiter=',')
	# return np.array(localizations), min_value
	return localizations, min_value