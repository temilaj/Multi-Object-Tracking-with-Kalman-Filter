import numpy as np
import cv2

bg_subtractor = cv2.createBackgroundSubtractorMOG2()

def generate_cell_localizations():
	image_list = []
	localizations = []

	for i in range(265, 376):
		frame = cv2.imread(f"../CS585-Cells/t1{i}.tif", cv2.IMREAD_COLOR)
		image_list.append(frame)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		fgmask = bg_subtractor.apply(gray)
		edges = cv2.Canny(fgmask, 50, 190, 3)

		# Retain only edges within the threshold
		ret, thresh = cv2.threshold(edges, 174, 255, 0)
		# Find contours
		contours, hierarchy = cv2.findContours(thresh,
													cv2.RETR_EXTERNAL,
													cv2.CHAIN_APPROX_SIMPLE)

		centers = [] 
		
		blob_radius_thresh = 10

		for cnt in contours:
			try:
				(x, y), radius = cv2.minEnclosingCircle(cnt)
				centeroid = (int(x), int(y))
				radius = int(radius)
				if (radius > blob_radius_thresh):
					cv2.circle(frame, centeroid, 6, (0, 255, 0), -1)
					b = np.array([[x], [y]])
					centers.append(np.round(b))
			except ZeroDivisionError:
				pass

		localizations.append(centers)
	return image_list, localizations
