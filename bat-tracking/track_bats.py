import numpy as np 
import cv2
from tracker import Tracker
import time
import imageio
import random

from bats import fetch_bat_localizations, generate_bat_image_list

bat_localizations_root_path = '../Localization'
segmentations_root_path = '../Segmentation'

# @njit
def format_localization_data(localizations, min_value):
	formated_localizations=[]
	for i in range(min_value):
		formated_localizations.append([])
		for j in range(len(localizations)):
			formated_localizations[i].append(localizations[j][i])
	return formated_localizations


bat_images = generate_bat_image_list()
bat_localizations, bat_min_value = fetch_bat_localizations(bat_localizations_root_path)
bat_localizations = np.array(format_localization_data(bat_localizations, bat_min_value))

def generate_random_colors(count):
	return  [(random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)) for i in range(count)]

def track(data, images, distinct_objects):
	image_tracks = []
	tracker = Tracker(150, 30, 5)
	
	track_colors = generate_random_colors(distinct_objects)

	for i in range(data.shape[1]):
		centers = data[:,i,:]
		frame = images[i]
		if (len(centers) > 0):
			tracker.update(centers)
			for j in range(len(tracker.tracks)):
				if (len(tracker.tracks[j].trace) > 1):
					x = int(tracker.tracks[j].trace[-1][0,0])
					y = int(tracker.tracks[j].trace[-1][0,1])
					tl = (x-10,y-10)
					br = (x+10,y+10)
					cv2.rectangle(frame,tl,br,track_colors[j],1)
					cv2.putText(frame,str(tracker.tracks[j].trackId), (x-10,y-20),0, 0.5, track_colors[j],2)
					for k in range(len(tracker.tracks[j].trace)):
						x = int(tracker.tracks[j].trace[k][0,0])
						y = int(tracker.tracks[j].trace[k][0,1])
						cv2.circle(frame,(x,y), 3, track_colors[j],-1)
					cv2.circle(frame,(x,y), 6, track_colors[j],-1)
				cv2.circle(frame,(int(data[j,i,0]),int(data[j,i,1])), 6, (0,0,0),-1)
			cv2.imshow('object tracking',frame)
			cv2.imwrite("../output/image"+str(i)+".jpg", frame)
			image_tracks.append(imageio.imread("../output/image"+str(i)+".jpg"))
			time.sleep(0.1) #slow down frame rate
			if cv2.waitKey(1) & 0xFF == ord('q'):
				cv2.destroyAllWindows()
				break

	print('saving generated gif image')
	imageio.mimsave('../output/Multi-Object-Tracking.gif', image_tracks, duration=0.08)
			
																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																										  

if __name__ == '__main__':
	track(bat_localizations, bat_images, bat_min_value)