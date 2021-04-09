import cv2
from cells import generate_cell_localizations
from tracker import Tracker
import random
import time
import imageio

segmentations_root_path = '../Segmentation'

def generate_random_colors(count):
	return  [(random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)) for i in range(count)]

cell_images, cell_localizations = generate_cell_localizations()

def track_object(images, localizations, name):
	image_tracks = []
	tracker = Tracker(150, 30, 5, 100)

	track_colors = generate_random_colors(9)

	# Infinite loop to process video frames
	for i in range(len(images)):
		frame = images[i]

		# obtain centeroids of the objects in the frame
		centers = localizations[i]
		if (len(centers) > 0):
			# Track object with the Kalman Filter
			tracker.update_tracks(centers)

			for j in range(len(tracker.tracks)):
				if (len(tracker.tracks[j].trace) > 1):
					x = int(tracker.tracks[j].trace[-1][0][0])
					y = int(tracker.tracks[j].trace[-1][1][0])
					# tl = (x-10,y-10)
					# br = (x+10,y+10)
					color = tracker.tracks[j].track_id % 9

					# cv2.rectangle(frame,tl,br,track_colors[color],1)
					cv2.putText(frame,str(tracker.tracks[j].track_id), (x-10,y-10),0, 0.5, track_colors[color],2)

					for k in range(len(tracker.tracks[j].trace)-1):
						x1 = int(tracker.tracks[j].trace[k][0][0])
						y1 = int(tracker.tracks[j].trace[k][1][0])
						x2 = int(tracker.tracks[j].trace[k+1][0][0])
						y2 = int(tracker.tracks[j].trace[k+1][1][0])

						cv2.line(frame, (x1, y1), (x2, y2),
								 track_colors[color], 2)

			cv2.imshow('Tracking', frame)
			cv2.imwrite(f"../output/{name}-image"+str(i)+".jpg", frame)
			image_tracks.append(imageio.imread(f"../output/{name}-image"+str(i)+".jpg"))
	
			
		# reduce frame rate
		cv2.waitKey(5)
		# time.sleep(0.1)

	cv2.destroyAllWindows()
	print('saving generated gif image')
	imageio.mimsave('../output/Multi-Object-Tracking.gif', image_tracks, duration=0.08)


if __name__ == "__main__":
	track_object(cell_images, cell_localizations, 'cells')