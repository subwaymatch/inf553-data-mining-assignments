import java.io.File
import java.nio.file._

import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}

object Task1 {
	def main(args: Array[String]): Unit = {
		// If input/output paths are not supplied, exit
		if (args.length < 2) {
			System.exit(1)
		}

		val ratingsFilePath = args(0)
		val outputFilePath = args(1)

		println("ratingsFilePath=" + ratingsFilePath)
		println("outputFilePath=" + outputFilePath)

		val outputFileObj: File = new File(outputFilePath)
		var outputTmpPath: Path = Paths.get(outputFileObj.getParent(), "tmp")

		val currentPath: Path = Paths.get(".").toAbsolutePath().normalize()

		// Create a new SparkSession instance
		val spark = SparkSession
				.builder()
				.master("local")
				.appName("Task01-Scala")
				.config("spark.sql.warehouse.dir", currentPath.toString())
				.getOrCreate()

		// Create DataFrame instance from ratings CSV file
		val ratings: DataFrame = spark.read
				.option("header", true)
				.option("inferSchema", true)
				.option("mode", "DROPMALFORMED")
				.csv(ratingsFilePath)

		// Group by movieId and average ratings as rating_avg
		val avg_ratings: DataFrame = ratings
				.groupBy("movieId")
				.agg(avg("rating").alias("rating_avg"))
				.orderBy(asc("movieId"))

		// Write to a single CSV file
		avg_ratings
				.repartition(1)
				.write
				.option("header", true)
				.option("sep", ",")
				.csv(outputTmpPath.toString())

		// Move the generated CSV file to our destination and remove temp directory
		AssignmentUtility.cleanUp(outputTmpPath, outputFileObj.toPath())
	}
}