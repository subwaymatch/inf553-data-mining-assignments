from __future__ import division
from pyspark import SparkConf, SparkContext
from collections import defaultdict
import time
import sys
import numpy as np
import itertools

numHashFuncs = 400
numBands = 40
numRowsPerBand = int(numHashFuncs / numBands)
primeArr = np.array([1, 227, 671, 2663, 3547, 6949, 10657, 17389, 32609, 59023])

print "numHashFuncs=" + str(numHashFuncs) + ", numBands=" + str(numBands) + ", numRowsPerBand=" + str(numRowsPerBand)

USER_INDEX = 0
MOVIE_INDEX = 1
RATING_INDEX = 2


def add_to_set(accum, new_val):
	accum += {new_val}


def hashRow(rowIndex, hashFuncIndex):
	return (3 * rowIndex + hashFuncIndex) % 671


def hashSetInBand(l, fromIndex, untilIndex):
	hashVal = np.dot(primeArr[:numRowsPerBand], l[fromIndex : untilIndex])
	hashVal = hashVal %  9066

	return hashVal


def computeJaccSimilarity(l1, l2):
	numIntersections = 0
	numUnions = 0
	arrLength = len(l1)

	for index in range(0, arrLength):
		if (l1[index] != 0) or (l2[index] != 0):
			numUnions += 1
		if (l1[index] == 1) and (l2[index] == 1):
			numIntersections += 1

	jaccSim = numIntersections / numUnions

	return jaccSim


def main():
	conf = SparkConf().setAppName("ModelBasedCF")\
		.setMaster("local")

	sc = SparkContext.getOrCreate(conf)
	sc.setLogLevel("ERROR")

	startTime = time.time()

	ratingsFilePath = sys.argv[1]

	data = sc.textFile(ratingsFilePath)
	dataHeader = data.first()

	# Ratings set as MatrixEntry ojects
	ratings = data.filter(lambda row: row != dataHeader)\
		.map(lambda row: row.split(","))\
		.map(lambda row: (int(row[USER_INDEX]), int(row[MOVIE_INDEX])))

	userIds = ratings.map(lambda r: r[USER_INDEX]).distinct().collect()
	userIds.sort()
	movieIds = ratings.map(lambda r: r[MOVIE_INDEX]).distinct().collect()
	movieIds.sort()

	numUsers = len(userIds)
	numMovies = len(movieIds)

	usersIndexMap = {}
	moviesIndexMap = {}
	moviesIndexLookupMap = {}

	for index, userId in enumerate(userIds):
		usersIndexMap[userId] = index

	for index, movieId in enumerate(movieIds):
		moviesIndexMap[movieId] = index
		moviesIndexLookupMap[index] = movieId

	candidatePairs = set()
	similarPairs = set()

	charMat = np.zeros((numMovies, numUsers), dtype=int)

	for r in ratings.collect():
		charMat[moviesIndexMap[r[MOVIE_INDEX]]][usersIndexMap[r[USER_INDEX]]] = 1

	print "Characteristic matrix generated"
	# print charMat

	signatureMat = np.full((numMovies, numHashFuncs), sys.maxint, dtype=int)

	print "Empty signatureMat generated"

	for hashIndex in range(numHashFuncs):
		for rowIndex in range(numUsers):
			for movieIndex in range(numMovies):
				if charMat[movieIndex][rowIndex] == 1:
					signatureMat[movieIndex][hashIndex] = min(signatureMat[movieIndex][hashIndex], hashRow(rowIndex, hashIndex))

	print "Signature matrix generated"
	# print signatureMat

	for bandIndex in range(numBands):
		fromIndex = bandIndex * numRowsPerBand
		untilIndex = (bandIndex + 1) * numRowsPerBand

		bandBucket = defaultdict(list)

		for movieIndex in range(numMovies):
			bucket = hashSetInBand(signatureMat[movieIndex], fromIndex, untilIndex)

			bandBucket[bucket].append(movieIndex)

		# Generate candidate pairs
		for _, moviesInBucket in bandBucket.iteritems():
			combinations = itertools.combinations(moviesInBucket, 2)

			for combination in combinations:
				combination = tuple(sorted(combination))
				candidatePairs.add(combination)

	print "candidatePairs size=" + str(len(candidatePairs))

	count = 0

	for pair in candidatePairs:
		jaccSimilarity = computeJaccSimilarity(charMat[pair[0]], charMat[pair[1]])

		if (count % 1000000 == 0):
			print "count=" + str(count)

		if jaccSimilarity >= 0.5:
			similarPairs.add((moviesIndexLookupMap[pair[0]], moviesIndexLookupMap[pair[1]], jaccSimilarity))

		count += 1

	print "similarPairs size=" + str(len(similarPairs))

	outputFileName = "YeJoo_Park_SimilarMovie_Jaccard.txt"
	printWriter = open(outputFileName, "a")

	similarPairsList = list(similarPairs)
	similarPairsList.sort()

	for pair in similarPairsList:
		printWriter.write(str(pair[0]) + ", " + str(pair[1]) + ", " + str(pair[2]))
		printWriter.write("\n")

	printWriter.close()

	print "--- %s seconds ---" % (time.time() - startTime)

main()