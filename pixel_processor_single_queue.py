import cv2
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import cv2
import colorsys

from work_handler import WorkHandler

def show_counters(counters):
	max_c = counters.max()
	counters = counters.astype(np.float64) / max_c

	i = 0
	while True:
		frame_resize = cv2.resize(counters[:,:,i % 256], 
					dsize = (1024, 1024))
		# cv2.imshow('Colors', counters[:,:,i % 256])
		cv2.imshow('Colors', frame_resize**.2)
		i += 1
		if cv2.waitKey(10) & 0xFF == ord('q'):
			break

def visualize_counters(counters):
	counters.shape = (256, 256, 256)
	hues = np.zeros(255, dtype = np.int64)
	for r in range(256):
		for g in range(256):
			for b in range(256):
				hsv = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
				hue = int(hsv[0] * 255)
				hues[hue] += counters[b, g, r]

	plt.plot(hues)
	plt.show()

if __name__ == "__main__":

	# print(cl.get_platforms())
	# print(cl.get_platforms()[0].get_devices())
	# print(cl.get_platforms()[1].get_devices())

	work_handler = WorkHandler(8)
	work_handler.manage_work()
	# image_loader = ImageLoader()
	# image_loader.load_all_images()

	# show_counters(work_handler.color_buffer)