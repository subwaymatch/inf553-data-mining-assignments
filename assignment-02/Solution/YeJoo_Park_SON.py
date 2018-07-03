from __future__ import division

import os
import sys
import time
import itertools

from pyspark import SparkContext
from collections import defaultdict
from math import floor


class SONAlgorithm:
	def __init__(self, case_number, csv_file_path, support, app_name, output_dir):
		self.support = support
		self.csv_file_path = csv_file_path
		self.frequent_items = None
		self.num_total_baskets = 0
		self.app_name = app_name

		# Switch key and value index of CSV file based on case number
		if case_number == 1:
			self.key_index = 0
			self.val_index = 1
		elif case_number == 2:
			self.key_index = 1
			self.val_index = 0
		else:
			print("Invalid case number parameter, it must either be 1 or 2")
			sys.exit()

		input_file_name, _ = os.path.splitext(os.path.basename(csv_file_path))

		output_file_name = self.app_name + "_" + input_file_name + ".case" + str(case_number) + "-" + str(self.support) + ".txt"
		self.output_file_path = os.path.join(output_dir, output_file_name)

	def start(self):
		f = open(self.output_file_path, "a+")

		ratings = self.get_ratings()

		# Get frequent singletons as a dict
		frequent_singletons = self.get_frequent_singletons(ratings)

		f_q_str = ", ".join(map(lambda v: "(" + str(v) + ")", sorted(frequent_singletons.keys())))
		f.write(f_q_str)
		f.write("\n\n")

		# Create baskets with only the items that are in frequent singletons dict
		# This utilizes the monotonicity
		baskets = ratings.filter(lambda row: row[1] in frequent_singletons) \
			.groupByKey() \
			.map(lambda row: set(row[1]))
		self.num_total_baskets = baskets.count()

		k = 2

		while True:
			candidates = baskets.mapPartitions(lambda chunk: self.find_candidates(chunk, k)).distinct().collect()

			candidate_counts = baskets.mapPartitions(lambda chunk: self.count_candidates(chunk, candidates))

			self.frequent_items = self.find_frequent_items(candidate_counts)

			frequent_set = set()

			for item_set in self.frequent_items:
				for item_id in item_set:
					frequent_set.add(item_id)

			baskets = baskets.map(lambda basket: basket.intersection(frequent_set))\
				.filter(lambda basket: len(basket) > 0)
			self.num_total_baskets = baskets.count()

			if len(self.frequent_items) == 0:
				f.close()
				break

			else:
				# Append to file
				frequent_items_str = str(sorted(self.frequent_items.keys())).strip("[]")
				f.write(frequent_items_str)
				f.write("\n\n")

			k += 1

	def get_ratings(self):
		# Local support threshold should be calculated as
		# (NUM_OF_BASKETS_IN_CHUNK) / (TOTAL_NUM_OF_BASKETS) * GLOBAL_SUPPORT
		sc = SparkContext.getOrCreate()

		# Read CSV file as rows
		ratings = sc.textFile(self.csv_file_path)

		# Remove header row
		ratings_header = ratings.first()
		ratings = ratings.filter(lambda row: row != ratings_header)

		# Split into separate columns, and extract (key, value) tuples
		# Discard number of stars and timestamp
		ratings = ratings.map(lambda row: row.split(",")) \
			.map(lambda row: (int(row[self.key_index]), int(row[self.val_index])))

		return ratings

	def get_frequent_singletons(self, ratings):
		return ratings.map(lambda row: (row[1], 1)) \
			.reduceByKey(lambda a, b: a + b) \
			.filter(lambda val: val[1] >= self.support) \
			.collectAsMap()

	def find_candidates(self, chunk_iterator, k):
		candidates = set()

		num_baskets_in_chunk = 0
		combination_count = defaultdict(int)

		for basket in chunk_iterator:
			num_baskets_in_chunk += 1
			combinations = itertools.combinations(basket, k)

			for combination in combinations:
				combination = tuple(sorted(combination))
				has_monotonic_subsets = True

				if k >= 3:
					if combination not in combination_count:
						subsets = itertools.combinations(combination, k - 1)
						for subset in subsets:
							subset = tuple(sorted(subset))
							if subset not in self.frequent_items:
								has_monotonic_subsets = False
								break

				if has_monotonic_subsets:
					combination_count[combination] += 1

		local_support_threshold = floor(self.support * (num_baskets_in_chunk / self.num_total_baskets))

		candidates_length = 0

		for id_set, count in combination_count.iteritems():
			if count >= local_support_threshold:
				candidates.add(id_set)
				candidates_length += 1

		return iter(candidates)

	@staticmethod
	def count_candidates(chunk_iterator, candidates):
		counts = []

		for basket in chunk_iterator:
			for candidate in candidates:
				if set(candidate).issubset(basket):
					counts.append((candidate, 1))

		return iter(counts)

	def find_frequent_items(self, candidates):
		return candidates.reduceByKey(lambda a, b: a + b)\
			.filter(lambda id_count: id_count[1] >= self.support)\
			.collectAsMap()


app_name = "YeJoo_Park_SON"
sparkContext = SparkContext(appName=app_name)
output_dir = "./"

# # Batch processing for submission
#
# sessions = [
# 	(1, "C:\\Users\\Park\\Downloads\\Assignment_02\\Data\\Small2.csv", 3),
# 	(2, "C:\\Users\\Park\\Downloads\\Assignment_02\\Data\\Small2.csv", 5),
# 	(1, "C:\\Users\\Park\\Downloads\\ml-latest-small\\ratings.csv", 120),
# 	(1, "C:\\Users\\Park\\Downloads\\ml-latest-small\\ratings.csv", 150),
# 	(2, "C:\\Users\\Park\\Downloads\\ml-latest-small\\ratings.csv", 180),
# 	(2, "C:\\Users\\Park\\Downloads\\ml-latest-small\\ratings.csv", 200),
# 	(1, "C:\\Users\\Park\\Downloads\\ml-20m\\ratings.csv", 30000),
# 	(1, "C:\\Users\\Park\\Downloads\\ml-20m\\ratings.csv", 35000),
# 	(2, "C:\\Users\\Park\\Downloads\\ml-20m\\ratings.csv", 2800),
# 	(2, "C:\\Users\\Park\\Downloads\\ml-20m\\ratings.csv", 3000),
# ]
#
# for session in sessions:
# 	start_time = time.time()
# 	son = SONAlgorithm(session[0], session[1], session[2], app_name, output_dir)
# 	son.start()
#
# 	print(session)
# 	print("%s seconds" % (time.time() - start_time))
#
# sys.exit()

# If input/output paths were not passed, exit
if len(sys.argv) < 4:
	print("Not enough parameters - check your input and output path")
	sys.exit()

# Save input/output paths to variables
case_number = int(sys.argv[1])
csv_file_path = sys.argv[2]
support = int(sys.argv[3])

print("case_number=", case_number)
print("csv_file_path=", csv_file_path)
print("support=", support)

start_time = time.time()
son = SONAlgorithm(case_number, csv_file_path, support, app_name, output_dir)
son.start()
print("--- %s seconds ---" % (time.time() - start_time))