import sys
from pyspark import SparkConf, SparkContext
from pyspark.accumulators import AccumulatorParam
from pyspark.mllib.recommendation import ALS, Rating

USER_INDEX = 0
MOVIE_INDEX = 1
RATING_INDEX = 2

def add_to_set(accum, new_val):
	accum += {new_val}

def main():
	conf = SparkConf().setAppName("YeJoo_Park_task2_ModelBasedCF")\
		.setMaster("local")

	sc = SparkContext.getOrCreate(conf)
	sc.setLogLevel("ERROR")

	ratingsFilePath = sys.argv[1]
	testFilePath = sys.argv[2]

	data = sc.textFile(testFilePath)
	dataHeader = data.first()

	testingSet = set(data\
		.filter(lambda row: row != dataHeader)\
		.map(lambda r: r.split(","))\
		.map(lambda r: (int(r[USER_INDEX]), int(r[MOVIE_INDEX])))\
		.collect())

	# Load and parse the data
	data = sc.textFile(ratingsFilePath)
	dataHeader = data.first()

	trainRatings = data\
		.filter(lambda row: row != dataHeader)\
		.map(lambda r: r.split(","))\
		.map(lambda r: Rating(int(r[USER_INDEX]), int(r[MOVIE_INDEX]), float(r[RATING_INDEX])))

	print "ratings.count() before filter=" + str(trainRatings.count())

	testRatings = trainRatings.filter(lambda rating: (rating.user, rating.product) in testingSet)
	trainRatings = trainRatings.filter(lambda rating: (rating.user, rating.product) not in testingSet)

	print "testingSetRatings.count()=" + str(testRatings.count())
	print "ratings.count() after filter=" + str(trainRatings.count())

	rank = 10
	numIterations = 12
	lamb = 0.1
	model = ALS.train(trainRatings, rank, numIterations, lamb)

	print "Training complete"

	userProducts = testRatings.map(lambda rating: (rating.user, rating.product))
	predictions = model.predictAll(userProducts).map(lambda r: ((r[0], r[1]), r[2]))
	ratesAndPreds = testRatings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
	absDiffBuckets = ratesAndPreds.map(lambda r: int(abs(r[1][0] - r[1][1]))) \
		.map(lambda d: min(d, 4)).cache()
	RMSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean()

	# Write predictions to file
	outputFileName = "YeJoo_Park_ModelBasedCF.txt"
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

main()
