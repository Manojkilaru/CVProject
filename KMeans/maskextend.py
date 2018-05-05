import numpy as np
import sys
from PIL import Image
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances
from skimage import io, color
import math
from matplotlib import pyplot as plt

inputName = sys.argv[1]
maskName = sys.argv[2]
image = io.imread(inputName)
mask = io.imread(maskName)
imageW = len(image[0])#image.size[0]
imageH = len(image)#image.size[1]
datadim = 2
#	Initialise data vector with attribute r,g,b,x,y for each pixel
dataVector = np.ndarray(shape=(imageW * imageH, datadim), dtype=float)
#	Initialise vector that holds which cluster a pixel is currently in
pixelClusterAppartenance = np.ndarray(shape=(imageW * imageH), dtype=int)

#	Populate data vector with data from input image
#	dataVector has 5 fields: red, green, blue, x coord, y coord
#rgb = io.imread(filename)
lab = color.rgb2hsv(image)
#lab = color.rgb2lab(image)
for y in range(0, imageH):
      for x in range(0, imageW):
      	xy = (x, y)
      	rgb = image[y][x]#image.getpixel(xy)
      	dataVector[x + y * imageW, 0] = rgb[0]
      	dataVector[x + y * imageW, 1] = rgb[1]
      	#dataVector[x + y * imageW, 2] = 0#rgb[2]
      	#dataVector[x + y * imageW, 3] = x/10
      	#dataVector[x + y * imageW, 4] = y/10
#	Standarize the values of our features
dataVector_scaled = preprocessing.normalize(dataVector)
#	Set centers
minValue = np.amin(dataVector_scaled)
maxValue = np.amax(dataVector_scaled)
distances = []
points = []
oldpoints = []
K = int(sys.argv[3])
# np.set_printoptions(threshold=np.nan)

# print(mask)
for y in range(imageH):
	for x in range(imageW):
		if(mask[y][x] > 128):
			oldpoints.append(tuple([x,y]))
#print(len(oldpoints))
c = dataVector_scaled[int((imageH/2)*imageW+imageW/2)] #change this
for y in range(imageH):
	for x in range(imageW):
		a = dataVector_scaled[y*imageW+x]
		if(mask[y][x] > 128):
			oldpoints.append(tuple([x,y]))
		for k in range(1,K):
			if(mask[y][x] > 128 and x+k<imageW and mask[y][x+k]< 128):
				b = dataVector_scaled[y*imageW+x+k]
				d = euclidean_distances(a.reshape(1, -1), b.reshape(1, -1)) + euclidean_distances(a.reshape(1, -1), c.reshape(1, -1))
				distances.append(d[0][0])
				points.append(tuple([x+k,y]))
			if(mask[y][x] > 128 and x-k>=0 and mask[y][x-k]< 128):
				b = dataVector_scaled[y*imageW+x-k]
				d = euclidean_distances(a.reshape(1, -1), b.reshape(1, -1)) + euclidean_distances(a.reshape(1, -1), c.reshape(1, -1))
				distances.append(d[0][0])
				points.append(tuple([x-k,y]))
			# if(mask[y][x] > 128 and mask[y+k][x]< 128):
			# 	b = dataVector_scaled[(y+k)*imageW+x]
			# 	d = euclidean_distances(a.reshape(1, -1), b.reshape(1, -1)) + euclidean_distances(a.reshape(1, -1), c.reshape(1, -1))
			# 	distances.append(d[0][0])
			# 	points.append(tuple([x,y+k]))
			if(mask[y][x] > 128 and y-k>=0 and mask[y-k][x]< 128):
				b = dataVector_scaled[(y-k)*imageW+x]
				d = euclidean_distances(a.reshape(1, -1), b.reshape(1, -1)) + euclidean_distances(a.reshape(1, -1), c.reshape(1, -1))
				distances.append(d[0][0])
				points.append(tuple([x,y-k]))
# points = [x for _,x in sorted(zip(distances,points))]
# distances.sort()
# end = len(distances)
# print(end)
# #print(distances)
# for i in range(len(distances)):
# 	if(distances[i]>threshold):
# 		end = i
# 		break
# points = points[:end]
# print(len(points))
per = sys.argv[4]
points = [x for _,x in sorted(zip(distances,points))][:int(len(distances)*float(per))]
data = np.array(distances*100)
# bins = np.linspace(math.ceil(min(data)), 
#                math.floor(max(data)),
#                100) # fixed number of bins

# plt.xlim([min(data)-5, max(data)+5])

# plt.hist(data, bins=bins, alpha=0.5)
# plt.title('Distances')
# plt.xlabel('variable X (20 evenly spaced bins)')
# plt.ylabel('count')

# plt.show()			

#newmask = mask*0
for p in points:
	x = p[0]
	y = p[1]
	mask[y][x] = 255
# for p in oldpoints:
# 	x = p[0]
# 	y = p[1]
# 	mask[y][x] = 1

# print(len(points))
io.imsave("../project/out1.jpg",mask)
# for y in range(imageH):
# 	for x in range(imageW):
# 		if mask[y][x] > 128:
# 			img1.putpixel((x, y), 1)
# 		else:
# 			img1.putpixel((x, y), 0)
# for p in points:
# 	x = p[0]
# 	y = p[1]
# 	img1.putpixel((x, y), 255)
# mask = io.imread(maskName)
# img1.save('newmask.jpg')
# newmask = io.imread("newmask.jpg")
#io.imsave("newmask.jpg",mathask+newmask)

# bins = np.linspace(math.ceil(min(data)), 
#                math.floor(max(data)),
#                100) # fixed number of bins

# plt.xlim([min(data)-5, max(data)+5])

# plt.hist(data, bins=bins, alpha=0.5)
# plt.title('Distances')
# plt.xlabel('variable X (20 evenly spaced bins)')
# plt.ylabel('count')

# plt.show()			
