from __future__ import division

import sys
import numpy as np

from pyspark import SparkConf, SparkContext

USER_INDEX = 0
MOVIE_INDEX = 1
RATING_INDEX = 2
RATING_MIN_VAL = 0.0
RATING_MAX_VAL = 5.0

def getKNNsAndSims(movieId, neighborIds, k=3):
	candSimilarities = []
	simRow = simMat[movieId]

	for nId in neighborIds:
		candSimilarities.append(simRow[nId])

	simOrder = np.argsort(candSimilarities)
	knnIndices = simOrder[-k - 1: -1]
	knnIdsAndSims =  []

	for knnIndex in knnIndices:
		knnIdsAndSims.append((neighborIds[knnIndex], candSimilarities[knnIndex]))

	return knnIdsAndSims


def predict(origUserId, origMovieId):
	userId = userIndexMap[origUserId]
	movieId = movieIndexMap[origMovieId]

	# Get a list of movies who have been rated by the same user
	ratedMovieOrigIds = list(ratedMoviesByUser[origUserId])
	ratedMovieIds = [movieIndexMap[origId] for origId in ratedMovieOrigIds]

	knnIdsAndSims = getKNNsAndSims(movieId, ratedMovieIds, 3)

	predictedRating = 0
	simSum = 0

	for nId, sim in knnIdsAndSims:
		predictedRating += sim * movieUserMat[nId][userId]
		simSum += sim

	if simSum != 0:
		predictedRating /= simSum
		predictedRating += movieAverages[origMovieId]

		predictedRating = min(predictedRating, RATING_MAX_VAL)
		predictedRating = max(RATING_MIN_VAL, predictedRating)

	else:
		predictedRating = movieAverages[origMovieId]

	return predictedRating

conf = SparkConf().setAppName("YeJoo_Park_task2_ItemBasedCF")\
	.setMaster("local")

sc = SparkContext.getOrCreate(conf)
sc.setLogLevel("ERROR")

ratingsFilePath = sys.argv[1]
testFilePath = sys.argv[2]
movieSimilarityFilePath = sys.argv[3]

data = sc.textFile(testFilePath)
dataHeader = data.first()

testingSet = set(data\
	.filter(lambda row: row != dataHeader) \
	.map(lambda r: r.split(",")) \
	.map(lambda r: (int(r[USER_INDEX]), int(r[MOVIE_INDEX])))\
	.collect())

print "testingSet size=" + str(len(testingSet))

data = sc.textFile(ratingsFilePath)
dataHeader = data.first()

ratings = data.filter(lambda r: r != dataHeader)\
	.map(lambda r: r.split(","))\
	.map(lambda r: (int(r[USER_INDEX]), int(r[MOVIE_INDEX]), float(r[RATING_INDEX])))

# Split ratings to training/test sets
testRatings = ratings.filter(lambda r: (r[USER_INDEX], r[MOVIE_INDEX]) in testingSet)
trainRatings = ratings.filter(lambda r: (r[USER_INDEX], r[MOVIE_INDEX]) not in testingSet)

print "testRatings size=" + str(testRatings.count())
print "trainRatings size=" + str(trainRatings.count())

# Find average values by user key
movieAverages = ratings\
	.map(lambda r: (r[MOVIE_INDEX], r[RATING_INDEX]))\
	.mapValues(lambda v: (v, 1)) \
	.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])) \
	.mapValues(lambda v: v[0] / v[1])\
	.collectAsMap()

# Maps for faster user: movies and movie: users lookup
ratedMoviesByUser = ratings.map(lambda r: (r[USER_INDEX], r[MOVIE_INDEX]))\
	.groupByKey()\
	.map(lambda r: (r[0], set(r[1])))\
	.collectAsMap()

ratedUsersByMovie = ratings.map(lambda r: (r[MOVIE_INDEX], r[USER_INDEX]))\
	.groupByKey()\
	.map(lambda r: (r[0], set(r[1])))\
	.collectAsMap()

# Normalize ratings
ratings = ratings.map(lambda r: (r[USER_INDEX], r[MOVIE_INDEX], r[RATING_INDEX] - movieAverages[r[MOVIE_INDEX]]))

# Extract unique users and movies
userIds = ratings.map(lambda r: r[0]).distinct().collect()
userIds.sort()
movieIds = ratings.map(lambda r: r[1]).distinct().collect()
movieIds.sort()

userIndexMap = {}
movieIndexMap = {}

for index, userId in enumerate(userIds):
	userIndexMap[userId] = index

for index, movieId in enumerate(movieIds):
	movieIndexMap[movieId] = index

numUsers = len(userIds)
numMovies = len(movieIds)

movieUserMat = np.zeros((numMovies, numUsers), dtype=float)

for r in ratings.collect():
	movieUserMat[movieIndexMap[r[MOVIE_INDEX]]][userIndexMap[r[USER_INDEX]]] = r[RATING_INDEX]

# print userItemMat

simMat = np.zeros((numMovies, numMovies), dtype=float)

for movieIndex in range(numUsers):
	simMat[movieIndex][movieIndex] = 1

data = sc.textFile(movieSimilarityFilePath)

similarityData = data.map(lambda l: l.split(","))\
	.map(lambda l: (int(l[0]), int(l[1]), float(l[2])))\
	.collect()

for index1, index2, similarity in similarityData:
	simMat[movieIndexMap[index1]][movieIndexMap[index2]] = similarity
	simMat[movieIndexMap[index2]][movieIndexMap[index1]] = similarity

# print simMat


# Make predictions based on KNNs of Cosine similarity (Pearson correlation)
ratesAndPreds = testRatings.map(lambda r: ((r[0], r[1]), (r[2], predict(r[0], r[1]))))
absDiffBuckets = ratesAndPreds.map(lambda r: int(abs(r[1][0] - r[1][1])))\
	.map(lambda d: min(d, 4)).cache()
RMSE  = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean()

# Write predictions to file
outputFileName = "YeJoo_Park_ItemBasedCF.txt"
printWriter = open(outputFileName, "a")

outputPreds = ratesAndPreds.map(lambda r: (r[0][0], r[0][1], r[1][1])).collect()
outputPreds.sort()

for pred in outputPreds:
	printWriter.write(str(pred[0]) + ", " + str(pred[1]) + ", " + str(pred[2]))
	printWriter.write("\n")

printWriter.close()

print ">=0 and <1: " + str(absDiffBuckets.filter(lambda d: d == 0).count())
print ">=1 and <2: " + str(absDiffBuckets.filter(lambda d: d == 1).count())
print ">=2 and <3: " + str(absDiffBuckets.filter(lambda d: d == 2).count())
print ">=3 and <4: " + str(absDiffBuckets.filter(lambda d: d == 3).count())
print ">=4: " + str(absDiffBuckets.filter(lambda d: d == 4).count())

print "RMSE=" + str(RMSE)