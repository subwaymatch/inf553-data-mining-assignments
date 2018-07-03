import sys
import os
import shutil

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# If input/output paths were not passed, exit
if len(sys.argv) < 3:
	print("Not enough parameters - check your input and output path")
	sys.exit()

# Save input/output paths to variables
input_path = sys.argv[1]
output_path = sys.argv[2]

(tmp_output_dir, output_file_name) = os.path.split(output_path)

# Append tmp to output directory
tmp_output_dir = os.path.join(tmp_output_dir, 'tmp')

# Build a SparkSession object
spark = SparkSession.builder.appName('Task1').getOrCreate()

# Read CSV and infer schema
ratings_df = spark.read.csv(input_path, mode="DROPMALFORMED", inferSchema=True, header=True)

# Calculate average ratings grouped by movieId
avg_rating_df = ratings_df\
	.groupBy('movieId')\
	.agg(F.mean('rating').alias('rating_avg'))\
	.orderBy(F.asc('movieId'))

# Write to CSV in tmp directory
avg_rating_df\
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