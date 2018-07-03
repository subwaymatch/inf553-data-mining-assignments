import sys
import os
import shutil

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

if len(sys.argv) < 4:
	print("Not enough parameters - check your input and output path")
	sys.exit()

# Input/output paths
ratings_path = sys.argv[1]
tags_path = sys.argv[2]
output_path = sys.argv[3]

(tmp_output_dir, output_file_name) = os.path.split(output_path)

# Append tmp to output directory
tmp_output_dir = os.path.join(tmp_output_dir, 'tmp')

# Build a SparkSession object
spark = SparkSession.builder.appName('Task2').getOrCreate()


ratings = spark.read.csv(ratings_path, mode="DROPMALFORMED", inferSchema=True, header=True)
tags = spark.read.csv(tags_path, mode="DROPMALFORMED", inferSchema=True, header=True)\
	.drop('timestamp')

# Inner join
ratings_joined = ratings.join(tags, 'movieId')

# Group by tags and calculate average rating of each tag
ratings_joined = ratings_joined\
	.groupBy('tag')\
	.agg(F.mean('rating').alias('rating_avg'))\
	.orderBy(F.desc('tag'))

# Write to CSV in tmp directory
ratings_joined\
	.repartition(1)\
	.write\
	.option('header', True)\
	.option('sep', ',')\
	.csv(tmp_output_dir)

# Find output CSV file
temp_csv_file = [filename for filename in os.listdir(tmp_output_dir) if filename.startswith('part-0000')][0]

# Move and rename the output CSV file and delete temp directory
shutil.move(os.path.join(tmp_output_dir, temp_csv_file), output_path)
shutil.rmtree(tmp_output_dir)