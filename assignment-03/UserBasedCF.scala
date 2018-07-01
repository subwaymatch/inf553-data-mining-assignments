import java.io.{BufferedWriter, FileWriter, PrintWriter}

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.Map
import scala.collection.mutable.ListBuffer

object UserBasedCF {
	val USER_INDEX = 0
	val MOVIE_INDEX = 1
	val RATING_INDEX = 2
	val RATING_MIN_VAL = 0.0
	val RATING_MAX_VAL = 5.0

	var userItemMat: Array[Array[Double]] = null
	var simMat: Array[Array[Double]] = null
	var userIndexMap: Map[Int, Int] = null
	var movieIndexMap: Map[Int, Int] = null
	var ratedMoviesByUser: Map[Int, Set[Int]] = null
	var ratedUsersByMovie: Map[Int, Set[Int]] = null
	var userAverages: Map[Int, Double] = null

	def dotProduct(vec1: Array[Double], vec2: Array[Double]): Double = {
		(vec1, vec2).zipped.map(_ * _).sum
	}

	def l2norm(vec: Array[Double]): Double = {
		Math.sqrt(dotProduct(vec, vec))
	}

	def cosSimilarity(vec1: Array[Double], l2vec1: Double, vec2: Array[Double], l2vec2: Double): Double = {
		dotProduct(vec1, vec2) / (l2vec1 * l2vec2)
	}

	def getKNNsAndSims(userId: Int, neighborIds: Seq[Int], k: Int = 5): Seq[(Int, Double)] = {
		val candSimilarities: ListBuffer[Double] = ListBuffer[Double]()
		val simRow: Array[Double] = simMat(userId)

		for (nId <- neighborIds) {
			candSimilarities.append(simRow(nId))
		}

		val simOrder: ListBuffer[Int] = candSimilarities.zipWithIndex.sortBy(e => (e._1)).map(e => e._2)
		var knnIndices = simOrder.takeRight(k + 1).dropRight(1)
		val knnIdsAndSims: ListBuffer[(Int, Double)] = ListBuffer[(Int, Double)]()

		for (knnIndex <- knnIndices) {
			knnIdsAndSims.append((neighborIds(knnIndex), candSimilarities(knnIndex)))
		}

		knnIdsAndSims
	}

	def predict(origUserId: Int, origMovieId: Int): Double = {
		val userId = userIndexMap(origUserId)
		val movieId = movieIndexMap(origMovieId)

		// Get a list of users who have rated the same movie
		val ratedUserOrigIds: Set[Int] = ratedUsersByMovie(origMovieId)
		val ratedUserIds: Seq[Int] = ratedUserOrigIds.map(origId => userIndexMap(origId)).toSeq

		val knnIdsAndSims: Seq[(Int, Double)] = getKNNsAndSims(userId, ratedUserIds, 5)

		var predictedRating: Double = 0.0
		var simSum: Double = 0.0

		knnIdsAndSims.foreach {
			case (nId, sim) => {
				predictedRating += sim * userItemMat(nId)(movieId)
				simSum += sim
			}
		}

		if (simSum != 0) {
			predictedRating /= simSum
			predictedRating += userAverages(origUserId)

			predictedRating = Math.min(predictedRating, RATING_MAX_VAL)
			predictedRating = Math.max(RATING_MIN_VAL, predictedRating)
		}

		else {
			predictedRating = userAverages(origUserId)
		}

		predictedRating
	}

	def main(args: Array[String]): Unit = {
		val conf = new SparkConf().setAppName("YeJoo_Park_task2_UserBasedCF")
				.setMaster("local")
		val sc = SparkContext.getOrCreate(conf)
		sc.setLogLevel("ERROR")

		val ratingsFilePath = args(0)
		val testFilePath = args(1)

		var data = sc.textFile(testFilePath)
		var dataHeader = data.first()

		val testingSet: Set[(Int, Int)] = data.filter(row => row != dataHeader)
				.map(r => r.split(","))
				.map(r => (r(USER_INDEX).toInt, r(MOVIE_INDEX).toInt))
				.collect().toSet

		println("testSet.size=" + testingSet.size)

		data = sc.textFile(ratingsFilePath)
		dataHeader = data.first()

		var ratings = data.filter(r => r != dataHeader)
				.map(r => r.split(","))
				.map(r => (r(USER_INDEX).toInt, r(MOVIE_INDEX).toInt, r(RATING_INDEX).toDouble))

		// Split ratings to training/test sets
		var testRatings = ratings.filter(r => testingSet.contains(r._1, r._2))
		var trainRatings = ratings.filter(r => !testingSet.contains(r._1, r._2))

		println("testRatings.size=" + testRatings.count())
		println("trainRatings.size=" + trainRatings.count())

		// Find average values by user key
		userAverages = trainRatings
				.map(r => (r._1, r._3))
				.mapValues(v => (v, 1))
				.reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2))
				.mapValues(v => v._1 / v._2)
				.collectAsMap()

		// Maps for faster user => movies and movie => users lookup
		ratedMoviesByUser = trainRatings.map(r => (r._1, r._2))
				.groupByKey()
				.map(r => (r._1, r._2.toSet))
				.collectAsMap()

		ratedUsersByMovie = trainRatings.map(r => (r._2, r._1))
				.groupByKey()
				.map(r => (r._1, r._2.toSet))
				.collectAsMap()

		// Normalize ratings
		trainRatings = trainRatings.map(r => (r._1, r._2, r._3 - userAverages(r._1)))

		// Extract unique users and movies
		val userIds: Set[Int] = trainRatings.map(r => r._1).distinct().collect().toSet
		val movieIds: Set[Int] = trainRatings.map(r => r._2).distinct().collect().toSet

		userIndexMap = userIds.zipWithIndex.toMap[Int, Int]
		movieIndexMap = movieIds.zipWithIndex.toMap[Int, Int]

		val numUsers: Int = userIds.size
		val numMovies: Int = movieIds.size

		println("numUsers=" + numUsers + ", numMovies=" + numMovies)

		userItemMat = Array.ofDim[Double](numUsers, numMovies)

		trainRatings.collect().foreach(r => {
			userItemMat(userIndexMap(r._1))(movieIndexMap(r._2)) = r._3
		})

		val l2norms: ListBuffer[Double] = ListBuffer[Double]()
		simMat = Array.ofDim[Double](numUsers, numUsers)

		for (userIndex <- 0 until numUsers) {
			l2norms.append(l2norm(userItemMat(userIndex)))
			simMat(userIndex)(userIndex) = 1
		}

		(0 until numUsers).combinations(2).foreach(pair => {
			val row1 = pair(0)
			val row2 = pair(1)

			simMat(row1)(row2) = cosSimilarity(userItemMat(row1), l2norms(row1), userItemMat(row2), l2norms(row2))
			simMat(row2)(row1) = simMat(row1)(row2)
		})

		// Exclude test sets of (userId, movieId) where the user or movie doesn't exist
		testRatings = testRatings.filter(r => userIds.contains(r._1) && movieIds.contains(r._2))

		println("testRatings size after excluding items=" + testRatings.count())

		// Make predictions based on KNNs of Cosine similarity (Pearson correlation)
		val ratesAndPreds: RDD[((Int, Int), (Double, Double))] = testRatings.map(r => ((r._1, r._2), (r._3, predict(r._1, r._2))))
		val absDiffBuckets: RDD[Int] = ratesAndPreds.map(r => Math.abs(r._2._1 - r._2._2).toInt)
				.map(d => Math.min(d, 4)).cache()
		val RMSE: Double = Math.sqrt(ratesAndPreds.map(r => Math.pow(r._2._1 - r._2._2, 2)).mean())

		// Write predictions to file
		val outputFileName: String = "YeJoo_Park_UserBasedCF.txt"
		val printWriter: PrintWriter = new PrintWriter(new BufferedWriter(new FileWriter(outputFileName)))

		val outputPreds: Array[(Int, Int, Double)] = ratesAndPreds.map(r => (r._1._1, r._1._2, r._2._2)).collect().sorted

		for (pred <- outputPreds) {
			printWriter.println(pred._1 + ", " + pred._2 + ", " + pred._3)
		}

		printWriter.close()

		println(">=0 and <1: " + absDiffBuckets.filter(d => d == 0).count())
		println(">=1 and <2: " + absDiffBuckets.filter(d => d == 1).count())
		println(">=2 and <3: " + absDiffBuckets.filter(d => d == 2).count())
		println(">=3 and <4: " + absDiffBuckets.filter(d => d == 3).count())
		println(">=4: " + absDiffBuckets.filter(d => d == 4).count())

		println("Root Mean Squared Error = " + RMSE)
	}
}
