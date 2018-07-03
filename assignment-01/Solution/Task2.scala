import java.io.File
import java.nio.file._

import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}

object Task2 {
	def main(args: Array[String]): Unit = {
		// If input/output paths are not supplied, exit
		if (args.length < 3) {
			System.exit(1)
		}

		val ratingsFilePath = args(0)
		val tagsFilePath = args(1)
		val outputFilePath = args(2)

		println("ratingsFilePath=" + ratingsFilePath)
		println("tagsFilePath=" + tagsFilePath)
		println("outputFilePath=" + outputFilePath)

		val outputFileObj: File = new File(outputFilePath)
		var outputTmpPath: Path = Paths.get(outputFileObj.getParent(), "tmp")

		val currentPath: Path = Paths.get(".").toAbsolutePath().normalize()

		// Create a new SparkSession instance
		val spark = SparkSession
				.builder()
				.master("local")
				.appName("Task02-Scala")
				.config("spark.sql.warehouse.dir", currentPath.toString())
				.getOrCreate()

		// Create DataFrame instance from ratings CSV file
		val ratings: DataFrame = spark.read
				.option("header", true)
				.option("inferSchema", true)
				.option("mode", "DROPMALFORMED")
				.csv(ratingsFilePath)

		ratings.show()

		// Create DataFrame instance from ratings CSV file
		val tags: DataFrame = spark.read
				.option("header", true)
				.option("inferSchema", true)
				.option("mode", "DROPMALFORMED")
				.csv(tagsFilePath)

		tags.show()

		var ratings_joined: DataFrame = ratings.join(tags, "movieId")

		ratings_joined.show()

		// Group by movieId and average ratings as rating_avg
		ratings_joined = ratings_joined
				.groupBy("tag")
				.agg(avg("rating").alias("rating_avg"))
				.orderBy(desc("tag"))

		// Write to a single CSV file
		ratings_joined
				.repartition(1)
				.write
				.option("header", true)
				.option("sep", ",")
				.csv(outputTmpPath.toString())

		// Move the generated CSV file to our destination and remove temp directory
		AssignmentUtility.cleanUp(outputTmpPath, outputFileObj.toPath())
	}
}