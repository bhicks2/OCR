import numpy as np 
import matplotlib.pyplot as plt
from sklearn.neighbors import DistanceMetric, KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.optimize import linear_sum_assignment
from old_baseline import Baseline
import canny, shape

# Gets the shape context distance for kNN algorithm
# Error: currently incompatible?
def mydist(x, y):
	h1 = np.reshape(x, (NUM_PTS, len(x)/NUM_PTS))
	h2 = np.reshape(y, (NUM_PTS, len(y)/NUM_PTS))
	cost = np.zeros((NUM_PTS, NUM_PTS))
	for i in xrange(NUM_PTS):
		for j in xrange(NUM_PTS):
			cost[i,j] = np.linalg.norm(h1[i] - h2[j])
	row_ind, col_ind = linear_sum_assignment(np.array(cost))
	return np.sum(np.linalg.norm(h1[row_ind] - h2[col_ind], axis=1))
dist = DistanceMetric.get_metric('pyfunc', func=mydist)

#################### Our data set #####################
n_neighbors = range(1, 11)
NUM_PTS = 100
RBINS = 5
ABINS = 12

baseline = Baseline()
baseline.readTrainingData('./training_examples2', binarize=False)

trainimg, trainlbl = baseline.data, baseline.classes
trainimg = np.reshape(trainimg, (len(trainimg), 100, 100))
trainhist = np.zeros((len(trainimg), NUM_PTS*RBINS*ABINS))
for i, img in enumerate(trainimg):
	edges = canny.canny(img, sigma=0.8, low_thresh=0.01, high_thresh=0.1)
	edges_sample = edges[np.sort(np.random.choice(edges.shape[0], (NUM_PTS,), replace=False))]
	shape_hist = shape.build_histograms(edges_sample, RBINS, ABINS, np.sqrt(img.shape[0]**2 + img.shape[1]**2))
	trainhist[i] = shape_hist.flatten()
print "Finished building histograms"

for n in n_neighbors:
	knn = KNeighborsClassifier(n, weights='uniform', algorithm='ball_tree', metric='euclidean', n_jobs=1)
	average = []
	for _ in range(10):
		X_train, X_test, y_train, y_test = train_test_split(trainhist, trainlbl, test_size=0.1)
		knn.fit(X_train, y_train)
		prediction = knn.predict(X_test)
		average.append(accuracy_score(prediction, y_test))
	print "k = %d: " % n, sum(average)/len(average)

####### Baseline: Accuracy 0.94 with k = 8.
# baseline = Baseline()
# baseline.readTrainingData('./training_examples', binarize=False)

# trainimg, trainlbl = baseline.data, baseline.classes

# for n in n_neighbors:
# 	knn = neighbors.KNeighborsClassifier(n, weights='uniform')
# 	average = []
# 	for _ in range(10):
# 		X_train, X_test, y_train, y_test = train_test_split(trainimg, trainlbl, test_size=0.1)
# 		knn.fit(X_train, y_train)
# 		prediction = knn.predict(X_test)
# 		average.append(accuracy_score(prediction, y_test))
# 	print "k = %d: " % n, sum(average)/10

'''
################### MNIST Data Set ######################
n_neighbors = 20
numtrain = 40000
numtest = 500

# ndarray has shape (60000, 28, 28), i.e. 60000 training images of 28 x 28
trainimg = idx2numpy.convert_from_file('train-images.idx3-ubyte')
trainlbl = idx2numpy.convert_from_file('train-labels.idx1-ubyte')
trnimg = trainimg.reshape(trainimg.shape[0], trainimg.shape[1]*trainimg.shape[2])
trnimg = trnimg/float(np.max(trnimg))

testimg = idx2numpy.convert_from_file('t10k-images.idx3-ubyte')
testlbl = idx2numpy.convert_from_file('t10k-labels.idx1-ubyte')
tstimg = testimg.reshape(testimg.shape[0], testimg.shape[1]*testimg.shape[2])
tstimg = tstimg/float(np.max(tstimg))

knn = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
knn.fit(trnimg[:numtrain], trainlbl[:numtrain])
prediction = knn.predict(tstimg[:numtest])
print accuracy_score(prediction, testlbl[:numtest])
'''

'''
### Remove some figures
for filename in os.listdir('./training_examples2'):
    # process data
    if not (filename.endswith(".png") or filename.endswith(".jpg")):
        continue

    underscoreIndex = filename.find("_")
    periodIndex = filename.find('.')
    if int(filename[underscoreIndex+1:periodIndex]) > 40:
    	os.remove('./training_examples2/' + filename)
'''