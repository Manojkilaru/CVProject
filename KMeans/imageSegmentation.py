import numpy as np
import sys
from PIL import Image
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances
from skimage import io, color
iterations = 5

#	Parse command-line arguments
#	sets K, inputName & outputName
if len(sys.argv) < 4:
	print("Error: Insufficient arguments, imageSegmentation takes three arguments")
	sys.exit()
else:
	K = int(sys.argv[1])
	if K < 3:
		print("Error: K has to be greater than 2")
		sys.exit()
	inputName = sys.argv[2]
	outputName = sys.argv[3]

#	Open input image
image = io.imread(inputName)#Image.open(inputName)
#print(len(image))
#print(len(image[0]))
imageW = len(image[0])#image.size[0]
imageH = len(image)#image.size[1]

#	Initialise data vector with attribute r,g,b,x,y for each pixel
dataVector = np.ndarray(shape=(imageW * imageH, 5), dtype=float)
#	Initialise vector that holds which cluster a pixel is currently in
pixelClusterAppartenance = np.ndarray(shape=(imageW * imageH), dtype=int)

#	Populate data vector with data from input image
#	dataVector has 5 fields: red, green, blue, x coord, y coord
#rgb = io.imread(filename)
lab = color.rgb2hsv(image)#color.rgb2lab(image)
# print(lab)
for y in range(0, imageH):
      for x in range(0, imageW):
      	xy = (x, y)
      	rgb = image[y][x]#image.getpixel(xy)
      	dataVector[x + y * imageW, 0] = rgb[0]
      	dataVector[x + y * imageW, 1] = rgb[1]
      	dataVector[x + y * imageW, 2] = rgb[2]
      	dataVector[x + y * imageW, 3] = x/600
      	dataVector[x + y * imageW, 4] = y/400

#	Standarize the values of our features
dataVector_scaled = preprocessing.normalize(dataVector)

#	Set centers
minValue = np.amin(dataVector_scaled)
maxValue = np.amax(dataVector_scaled)

centers = np.ndarray(shape=(K,5))
for index, center in enumerate(centers):
	centers[index] = np.random.uniform(minValue, maxValue, 5)

for iteration in range(iterations):
	#	Set pixels to their cluster
	print("hi")
	for idx, data in enumerate(dataVector_scaled):
		distanceToCenters = np.ndarray(shape=(K))
		for index, center in enumerate(centers):
			distanceToCenters[index] = euclidean_distances(data.reshape(1, -1), center.reshape(1, -1))
		pixelClusterAppartenance[idx] = np.argmin(distanceToCenters)

	##################################################################################################
	#	Check if a cluster is ever empty, if so append a random datapoint to it
	clusterToCheck = np.arange(K)		#contains an array with all clusters
										#e.g for K=10, array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
	clustersEmpty = np.in1d(clusterToCheck, pixelClusterAppartenance)
										#^ [True True False True * n of clusters] False means empty
	for index, item in enumerate(clustersEmpty):
		if item == False:
			pixelClusterAppartenance[np.random.randint(len(pixelClusterAppartenance))] = index
			# ^ sets a random pixel to that cluster as mentioned in the homework writeup
	##################################################################################################

	#	Move centers to the centroid of their cluster
	for i in range(K):
		dataInCenter = []

		for index, item in enumerate(pixelClusterAppartenance):
			if item == i:
				dataInCenter.append(dataVector_scaled[index])
		dataInCenter = np.array(dataInCenter)
		centers[i] = np.mean(dataInCenter, axis=0)

	#TODO check for convergence
	# print("Centers Iteration num", iteration, ": \n", centers)

#	set the pixels on original image to be that of the pixel's cluster's centroid
for index, item in enumerate(pixelClusterAppartenance):
	dataVector[index][0] = int(round(centers[item][0] * 255))
	dataVector[index][1] = int(round(centers[item][1] * 255))
	dataVector[index][2] = int(round(centers[item][2] * 255))

#	Save image
image = Image.new("RGB", (imageW, imageH))

for y in range(imageH):
	for x in range(imageW):
	 	image.putpixel((x, y), (int(dataVector[y * imageW + x][0]), 
	 							int(dataVector[y * imageW + x][1]),
	 							int(dataVector[y * imageW + x][2])))
image.save("out5.jpg")


tempVector = np.ndarray(shape=(imageW * imageH))
for index, item in enumerate(pixelClusterAppartenance):
	tempVector[index] = item
mask = np.zeros(shape=(imageW,imageH),dtype=np.bool_)
mainItem = tempVector[int((imageH/2) * imageW + (imageW/2))]
img = Image.new("1", (imageW, imageH))
for y in range(imageH):
	for x in range(imageW):
		if mainItem == tempVector[y * imageW + x]:
			img.putpixel((x, y), 1)
		else:
			img.putpixel((x, y), 0)

img.save("out1.jpg")

# img = Image.new("1", (imageW, imageH))
# for y in range(imageH):
# 	for x in range(imageW):
# 		if mainItem == tempVector[y * imageW + x]:
# 			img.putpixel((x, y), 0)
# 		else:
# 			img.putpixel((x, y), 1)

# img.save("out2.jpg")