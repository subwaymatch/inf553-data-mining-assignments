import java.io.{File, PrintWriter}

import org.apache.spark.rdd.{RDD}
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.{Map, mutable}
import scala.collection.mutable.ArrayBuffer

import scala.math.Ordering.Implicits._

object FrequentItemsetsSON {
	private var caseNumber: Int = 0
	private var csvFilePath: String = ""
	private var support: Int = 0

	private var keyIndex: Int = 0
	private var valIndex: Int = 0

	private val appName: String = "YeJoo_Park_SON"
	private val outputDir: String = "./"
	private var outputFile: File = null
	private var printWriter: PrintWriter = null
	private var frequentItems: Map[Set[Int], Int] = null

	private var numTotalBaskets: Long = 0

	def init() {
		if (this.caseNumber == 1) {
			this.keyIndex = 0
			this.valIndex = 1
		} else if (this.caseNumber == 2) {
			this.keyIndex = 1
			this.valIndex = 0
		} else {
			println("Invalid case number parameter, it must either be 1 or 2")
			System.exit(1)
		}

		var inputFileName: String = new File(csvFilePath).getName()
		inputFileName = inputFileName.dropRight(inputFileName.length - inputFileName.lastIndexOf("."))
		val outputFileName: String = this.appName + "_" + inputFileName + ".case" + this.caseNumber + "-" + this.support + ".txt"
		this.outputFile = new File(outputDir, outputFileName)
	}

	def start(): Unit = {
		this.printWriter = new PrintWriter(this.outputFile)

		val ratings = getRatings()

		val frequentSingletons: Map[Int, Int] = getFrequentSingletons(ratings)
		val frequentSingletonStr: String = frequentSingletons.keySet.toList.sorted.map(x => "(" + x.toString + ")").mkString(", ")

		this.printWriter.println(frequentSingletonStr)
		this.printWriter.println()

		var baskets = ratings.filter(r => frequentSingletons.contains(r._2))
				.groupByKey()
				.map(r => r._2.toList.sorted.toSet)

		this.numTotalBaskets = baskets.count()

		var k: Int = 2
		var hasMoreK: Boolean = true

		while(hasMoreK) {
			val candidates = baskets.mapPartitions(chunk => findFrequentCandidates(chunk, k)).distinct().collect().toSet

			val candidateCounts = baskets.mapPartitions(chunk => countCandidates(chunk, candidates))

			this.frequentItems = findFrequentItems(candidateCounts)

			val frequentSet: mutable.Set[Int] = mutable.Set[Int]()

			for ((idSet, count) <- this.frequentItems) {
				idSet.foreach(id => frequentSet.add(id))
			}

			baskets = baskets.map(basket => basket.intersect(frequentSet))
					.filter(basket => basket.size > 0)

			this.numTotalBaskets = baskets.count()

			if (this.frequentItems.size == 0) {
				hasMoreK = false
			}

			else {
				val frequentItemsReordered: Iterable[List[Int]] = this.frequentItems.keySet.map(idSet => idSet.toList.sorted).toList.sorted

				printWriter.println(frequentItemsReordered.mkString(", ").replaceAll("List", ""))
				printWriter.println("")
			}

			k += 1
		}

		this.printWriter.close()
	}

	def getRatings(): RDD[(Int, Int)] = {
		val conf = new SparkConf().setAppName(this.appName)
        		.setMaster("local")
		val sc = SparkContext.getOrCreate(conf)
		sc.setLogLevel("ERROR")

		// Read CSV files as rows
		var ratingsStringRDD = sc.textFile(this.csvFilePath, 2)

		// Remove header row
		val ratings_header = ratingsStringRDD.first()
		ratingsStringRDD = ratingsStringRDD.filter(row => row != ratings_header)

		val ratings: RDD[(Int, Int)] = ratingsStringRDD.map(line => {
			val columnValues: Array[String] = line.split(",")
			(columnValues(this.keyIndex).toInt, columnValues(this.valIndex).toInt)
		})

		return ratings
	}

	def getFrequentSingletons(ratings: RDD[(Int, Int)]): Map[Int, Int] = {
		val frequentSingletons = ratings
        		.map(r => (r._2, 1))
				.reduceByKey((x, y) => (x + y))
        		.filter(r => r._2 >= this.support)
        		.collectAsMap()

		return frequentSingletons
	}

	def findFrequentCandidates(chunkIterator: Iterator[Set[Int]], k: Int): Iterator[Set[Int]] = {
		val candidates: ArrayBuffer[Set[Int]] = ArrayBuffer()

		var numBasketsInChunk: Int = 0
		val combinationCountMap: mutable.Map[Set[Int], Int] = mutable.Map[Set[Int], Int]()

		while (chunkIterator.hasNext) {
			val basket: Set[Int] = chunkIterator.next()

			numBasketsInChunk += 1
			val combinations = basket.toSeq.combinations(k)

			combinations.foreach(combination => {
				val combinationSet = combination.toSet
				var hasMonotonicSubsets: Boolean = true

				if (k >= 3) {
					if (!combinationCountMap.contains(combinationSet)) {
						val subsets = combination.combinations(k - 1)
						var continueSubsetCheck = true

						while (subsets.hasNext && continueSubsetCheck) {
							val subset = subsets.next()

							if (!this.frequentItems.contains(subset.toSet)) {
								hasMonotonicSubsets = false
								continueSubsetCheck = false
							}
						}
					}
				}

				if (hasMonotonicSubsets) {
					if (combinationCountMap.contains(combinationSet)) {
						combinationCountMap(combinationSet) += 1
					}

					else {
						combinationCountMap(combinationSet) = 1
					}
				}
			})
		}

		val localSupportThreshold: Double = Math.floor(this.support.toDouble * (numBasketsInChunk.toDouble / this.numTotalBaskets.toDouble))

		for ((idSet, count) <- combinationCountMap) {
			if (count >= localSupportThreshold) {
				candidates += idSet
			}
		}

		return candidates.toIterator
	}

	def countCandidates(chunkIterator: Iterator[Set[Int]], candidates: Set[Set[Int]]): Iterator[(Set[Int], Int)] = {
		val candidateCounts: ArrayBuffer[(Set[Int], Int)] = ArrayBuffer()

		while (chunkIterator.hasNext) {
			val basket: Set[Int] = chunkIterator.next()

			candidates.foreach(candidate => {
				if (candidate.subsetOf(basket)) {
					candidateCounts += Tuple2(candidate, 1)
				}
			})
		}

		return candidateCounts.toIterator
	}


	def findFrequentItems(candidates: RDD[(Set[Int], Int)]): Map[Set[Int], Int] = {
		val results = candidates.reduceByKey((a, b) => a + b)
			.filter(id_count => id_count._2 >= this.support)
			.collectAsMap()

		return results
	}

	def main(args: Array[String]): Unit = {
		println("Begin FrequentItemsetsSON execution")

		/*
		val sessions = ArrayBuffer[(Int, String, Int)]()

		sessions += Tuple3(1, "C:\\Users\\Park\\Downloads\\Assignment_02\\Data\\Small2.csv", 3)
		sessions += Tuple3(2, "C:\\Users\\Park\\Downloads\\Assignment_02\\Data\\Small2.csv", 5)
		sessions += Tuple3(1, "C:\\Users\\Park\\Downloads\\ml-latest-small\\ratings.csv", 120)
		sessions += Tuple3(1, "C:\\Users\\Park\\Downloads\\ml-latest-small\\ratings.csv", 150)
		sessions += Tuple3(2, "C:\\Users\\Park\\Downloads\\ml-latest-small\\ratings.csv", 180)
		sessions += Tuple3(2, "C:\\Users\\Park\\Downloads\\ml-latest-small\\ratings.csv", 200)
		sessions += Tuple3(1, "C:\\Users\\Park\\Downloads\\ml-20m\\ratings.csv", 30000)
		sessions += Tuple3(1, "C:\\Users\\Park\\Downloads\\ml-20m\\ratings.csv", 35000)
		sessions += Tuple3(2, "C:\\Users\\Park\\Downloads\\ml-20m\\ratings.csv", 2800)
		sessions += Tuple3(2, "C:\\Users\\Park\\Downloads\\ml-20m\\ratings.csv", 3000)

		sessions.foreach(session => {
			val startTime = System.currentTimeMillis()
			println(session)

			this.caseNumber = session._1
			this.csvFilePath = session._2
			this.support = session._3

			init()
			start()

			val endTime = System.currentTimeMillis()
			println("Time=", (endTime - startTime) / 1000)
		})

		if (sessions.length > 0) return
		*/

		this.caseNumber = args(0).toInt
		this.csvFilePath = args(1)
		this.support = args(2).toInt

		println("case_number=", this.caseNumber)
		println("csv_file_path=", this.csvFilePath)
		println("support=", this.support)

		val startTime = System.currentTimeMillis()

		init()
		start()

		val endTime = System.currentTimeMillis()

		println("Time=", (endTime - startTime) / 1000)
	}
}
