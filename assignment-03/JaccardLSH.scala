import java.io.{BufferedWriter, FileWriter, PrintWriter}

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

object JaccardLSH {
	val numHashFuncs = 400
	val numBands = 40
	val numRowsPerBand = numHashFuncs / numBands
	val primeArr: Array[Int] = Array[Int](1, 227, 671, 2663, 3547, 6949, 10657, 17389, 32609, 59023)

	def hashRow(rowIndex: Int, hashFuncIndex: Int): Int = {
		(3 * rowIndex + hashFuncIndex) % 671
	}

	def hashSetInBand(l: Array[Int], fromIndex: Int, untilIndex: Int): Int = {
		var hashVal = 0

		for (index <- fromIndex until untilIndex) {
			hashVal += l(index) * primeArr(index - fromIndex)
		}

		hashVal = hashVal % 9066

		return hashVal
	}

	def computeJaccSimilarity(l1: Array[Int], l2: Array[Int]): Double = {
		var numIntersections: Int = 0
		var numUnions: Int = 0

		for (index <- 0 until l1.length) {
			if (l1(index) != 0 || l2(index) != 0) numUnions += 1
			if (l1(index) == 1 && l2(index) == 1) numIntersections += 1
		}

		val jaccSim = numIntersections.toDouble / numUnions.toDouble

		return jaccSim
	}

	def main(args: Array[String]): Unit = {
		val conf = new SparkConf().setAppName("ModelBasedCF")
				.setMaster("local")
		val sc = SparkContext.getOrCreate(conf)
		sc.setLogLevel("ERROR")

		val startTime = System.currentTimeMillis()

		val ratingsFilePath = args(0)

		val data = sc.textFile(ratingsFilePath)
		val dataHeader = data.first()

		// Ratings set as MatrixEntry objects
		val ratings: RDD[(Int, Int)] = data
				.filter(row => row != dataHeader)
				.map(_.split(',') match {
					case Array(user, movie, rating, timestamp) =>
						(user.toInt, movie.toInt)
				})

		val userIdAccum = new SetAccumulator[Int]()
		val movieIdAccum = new SetAccumulator[Int]()

		sc.register(userIdAccum)
		sc.register(movieIdAccum)

		ratings.foreach(r => {
			userIdAccum.add(r._1)
			movieIdAccum.add(r._2)
		})

		val userIds: List[Int] = userIdAccum.value.toList.sorted
		val movieIds: List[Int] = movieIdAccum.value.toList.sorted

		val numUsers = userIds.size
		val numMovies = movieIds.size

		println("numUsers=" + numUsers)
		println("numMovies=" + numMovies)

		val usersIndexMap: Map[Int, Int] = userIds.zipWithIndex.toMap[Int, Int]
		val moviesIndexMap: Map[Int, Int] = movieIds.zipWithIndex.toMap[Int, Int]
		val moviesIndexLookupMap = for ((k, v) <- moviesIndexMap) yield (v, k)
		val candidatePairs: mutable.Set[(Int, Int)] = mutable.Set[(Int, Int)]()
		val similarPairs: mutable.Set[(Int, Int, Double)] = mutable.Set[(Int, Int, Double)]()

		// Create characteristic matrix
		val charMat: Array[Array[Int]] = Array.ofDim[Int](movieIds.size, userIds.size)
		ratings.collect().foreach(r => {
			charMat(moviesIndexMap(r._2))(usersIndexMap(r._1)) = 1
		})

		// Create signature matrix
		val signatureMat: Array[Array[Int]] = Array.fill[Int](numMovies, numHashFuncs)(Int.MaxValue)

		println("numHashFuncs=" + numHashFuncs + ", numBands=" + numBands + ", numRowsPerBand=" + numRowsPerBand)

		for (hashIndex <- 0 until numHashFuncs) {
			for (rowIndex <- 0 until numUsers) {
				for (movieIndex <- 0 until numMovies) {
					if (charMat(movieIndex)(rowIndex) == 1) {
						signatureMat(movieIndex)(hashIndex) = Math.min(signatureMat(movieIndex)(hashIndex), hashRow(rowIndex, hashIndex))
					}
				}
			}
		}

		for (bandIndex <- 0 until numBands) {
			val fromIndex = bandIndex * numRowsPerBand
			val untilIndex = (bandIndex + 1) * numRowsPerBand

			val bandBucket: scala.collection.mutable.Map[Int, ListBuffer[Int]] = scala.collection.mutable.Map[Int, ListBuffer[Int]]()

			for (movieIndex <- 0 until numMovies) {
				val bucket = hashSetInBand(signatureMat(movieIndex), fromIndex, untilIndex)

				if (!bandBucket.contains(bucket)) bandBucket(bucket) = new ListBuffer[Int]()

				bandBucket(bucket) += movieIndex
			}

			// Generate candidate pairs
			bandBucket.foreach(b => {
				val combinations = b._2.combinations(2)

				combinations.foreach(pair => {
					candidatePairs.add((pair(0), pair(1)))
				})
			})
		}

		println("candidatePairs.size=" + candidatePairs.size)

		var count = 0

		candidatePairs.foreach(pair => {
			val jaccSimilarity = computeJaccSimilarity(charMat(pair._1), charMat(pair._2))

			if (count % 1000000 == 0) {
				println("count=" + count)
			}

			if (jaccSimilarity >= 0.5) {
				similarPairs.add((moviesIndexLookupMap(pair._1), moviesIndexLookupMap(pair._2), jaccSimilarity))
			}

			count += 1
		})

		print("similairPairs.size=" + similarPairs.size)

		val outputFileName = "YeJoo_Park_SimilarMovie_Jaccard.txt"
		// val printWriter: PrintWriter = new PrintWriter(new File(outputFileName))
		val printWriter: PrintWriter = new PrintWriter(new BufferedWriter(new FileWriter(outputFileName)))


		similarPairs.toList.sorted.foreach(pair => {
			printWriter.println(pair._1 + ", " + pair._2 + ", " + pair._3)
		})

		printWriter.close()

		val endTime = System.currentTimeMillis()
		println("Time=", (endTime - startTime) / 1000 + " seconds")
	}
}
