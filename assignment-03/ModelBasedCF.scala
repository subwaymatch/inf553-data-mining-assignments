import java.io.{File, PrintWriter}

import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object ModelBasedCF {
	val RATING_MIN_VAL = 0.0
	val RATING_MAX_VAL = 5.0

	def main(args: Array[String]): Unit = {
		val conf = new SparkConf().setAppName("YeJoo_Park_task2_ModelBasedCF")
				.setMaster("local")
		val sc = SparkContext.getOrCreate(conf)
		sc.setLogLevel("ERROR")

		val ratingsFilePath = args(0)
		val testFilePath = args(1)

		var data = sc.textFile(testFilePath)
		var dataHeader = data.first()

		// Retrieve testing data set
		val testingData = data
				.filter(row => row != dataHeader)
				.map(_.split(',') match {
					case Array(user, item) =>
						(user.toInt, item.toInt)
				})

		// Retrieve the set
		val testingSet: Set[(Int, Int)] = testingData
				.collect()
				.toSet

		println("testingSet.size=" + testingSet.size)

		// Load and parse the data
		data = sc.textFile(ratingsFilePath)
		dataHeader = data.first()

		// Create ratings
		var trainRatings: RDD[Rating] = data
				.filter(row => row != dataHeader)
				.map(_.split(',') match {
					case Array(user, product, rate, _) =>
						Rating(user.toInt, product.toInt, rate.toDouble)
				})

		println("ratings.count() before filter=" + trainRatings.count())

		val testRatings = trainRatings.filter(rating => testingSet.contains((rating.user, rating.product)))
		trainRatings = trainRatings.filter(rating => !testingSet.contains((rating.user, rating.product)))

		println("testingSetRatings.count()=" + testRatings.count)
		println("ratings.count() after filter=" + trainRatings.count())

		// Build the recommendation model using ALS
		val rank = 10
		val numIterations = 12
		val lambda = 0.1
		val model = ALS.train(trainRatings, rank, numIterations, lambda)

		println("Training complete")

		// Evaluate the model on rating data
		val usersProducts = testRatings.map { case Rating(user, product, rate) =>
			(user, product)
		}
		val predictions: RDD[((Int, Int), Double)] =
			model.predict(usersProducts).map { case Rating(user, product, rate) =>
				((user, product), rate)
			}
					.map(r => (r._1, Math.max(r._2, RATING_MIN_VAL)))
					.map(r => (r._1, Math.min(r._2, RATING_MAX_VAL)))

		val ratesAndPreds = testRatings.map { case Rating(user, product, rate) =>
			((user, product), rate)
		}.join(predictions)

		val absDiffBuckets = ratesAndPreds.map(r => (Math.abs(r._2._1 - r._2._2)).toInt)
				.map(d => Math.min(d, 4)).cache()

		val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
			val err = (r1 - r2)
			err * err
		}.mean()

		println("rank=" + rank + ", numIterations=" + numIterations + ", lambda=" + lambda)

		val outputFileName = "YeJoo_Park_ModelBasedCF.txt"
		val printWriter: PrintWriter = new PrintWriter(new File(outputFileName))

		predictions.collect().toList.sorted.foreach(pred => {
			printWriter.println(pred._1._1.toString + ", " + pred._1._2.toString + ", " + pred._2.toString)
		})

		printWriter.close()

		println(">=0 and <1: " + absDiffBuckets.filter(d => d == 0).count())
		println(">=1 and <2: " + absDiffBuckets.filter(d => d == 1).count())
		println(">=2 and <3: " + absDiffBuckets.filter(d => d == 2).count())
		println(">=3 and <4: " + absDiffBuckets.filter(d => d == 3).count())
		println(">=4: " + absDiffBuckets.filter(d => d == 4).count())

		println("Root Mean Squared Error = " + Math.sqrt(MSE))
	}
}
