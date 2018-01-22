 
 
# ------------------------------------------------------------------------------
# Step 1:
# Creating an Pyspark dataframe from a hive table
# Importing the train data, the test data and the scoring data
# ------------------------------------------------------------------------------
 
data_train = sqlContext.sql("SELECT * FROM my_db.Sample_50pct_train")
data_test = sqlContext.sql("SELECT * FROM my_db.Sample_50pct_test")
data_score = sqlContext.sql("SELECT * FROM my_db.Sample_scoring")
 
 
# ------------------------------------------------------------------------------
# Type of object
# ------------------------------------------------------------------------------
type(data_train)
# <class 'pyspark.sql.dataframe.DataFrame'>
 
 
# ------------------------------------------------------------------------------
# Step 2:
# Converting to a pyspark RDD from pyspark Dataframe
# ------------------------------------------------------------------------------
 
data_train_rdd = data_train.rdd
data_test_rdd = data_test.rdd
data_score_rdd = data_score.rdd
type(data_train_rdd)
#<class 'pyspark.rdd.RDD'>
 
 
 
 
# ------------------------------------------------------------------------------
# Step 3:
# Importing libraries for converting the data frame to a dense vector
# We need to convert this Data Frame to an RDD of LabeledPoint.
# ------------------------------------------------------------------------------
 
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
 
# ------------------------------------------------------------------------------
# Step 4(a):
# For the train and test data, the structure is as follows:
#    The first row is Customer_ID; second row is the Y variable; and the third row onwards are the X Variables
# ------------------------------------------------------------------------------
 
transformed_train_df = data_train_rdd.map(lambda row: LabeledPoint(row[1], Vectors.dense(row[2:])))
transformed_test_df = data_test_rdd.map(lambda row: LabeledPoint(row[1], Vectors.dense(row[2:])))
 
 
# ------------------------------------------------------------------------------
# Step 4(b):
# For the scoring data, the structure is as follows:
#   The first row is the Customer_ID; and the second row onwards are the X Variables
# ------------------------------------------------------------------------------
 
transformed_score_df = data_score_rdd.map(lambda row: LabeledPoint(row[0], Vectors.dense(row[1:])))
 
 
# ------------------------------------------------------------------------------
# Step 5:
# Random Forest model
# ------------------------------------------------------------------------------
 
from pyspark.mllib.tree import RandomForest
 
# ------------------------------------------------------------------------------
# Step 5(a):
# Parameters for the Random Forest model
# ------------------------------------------------------------------------------
 
RANDOM_SEED = 10904
RF_NUM_TREES = 100
RF_MAX_DEPTH = 4
RF_MAX_BINS = 100
 
# ------------------------------------------------------------------------------
# Step 5(b):
# Training a Random Forest model on the dataset
# ------------------------------------------------------------------------------
 
model = RandomForest.trainClassifier(transformed_train_df, numClasses=2, categoricalFeaturesInfo={}, \
    numTrees=RF_NUM_TREES, featureSubsetStrategy="log2", impurity="entropy", \
    maxDepth=RF_MAX_DEPTH, maxBins=RF_MAX_BINS, seed=RANDOM_SEED)
 
# ------------------------------------------------------------------------------
# Step 5©:
# Make predictions and compute accuracy
# ------------------------------------------------------------------------------
 
predictions = model.predict(transformed_test_df.map(lambda x: x.features))
labels_and_predictions = transformed_test_df.map(lambda x: x.label).zip(predictions)
model_accuracy = labels_and_predictions.filter(lambda x: x[0] == x[1]).count() / float(transformed_test_df.count())
print("Model accuracy: %.3f%%" % (model_accuracy * 100))
 
 
# ------------------------------------------------------------------------------
# Step 5(d):
# Model evaluation
# ------------------------------------------------------------------------------
 
from pyspark.mllib.evaluation import BinaryClassificationMetrics
 
metrics = BinaryClassificationMetrics(labels_and_predictions)
print("Area under Precision/Recall (PR) curve: %.f" % (metrics.areaUnderPR * 100))
print("Area under Receiver Operating Characteristic (ROC) curve: %.3f" % (metrics.areaUnderROC * 100))
 
# ------------------------------------------------------------------------------
# Step 6:
# Scoring dataset
# ------------------------------------------------------------------------------
score_predictions = model.predict(transformed_score_df.map(lambda x: x.features))
score_labels_and_predictions = transformed_score_df.map(lambda x: x.label).zip(score_predictions)
 
# ------------------------------------------------------------------------------
# Step 7:
# Creating a final hive table for the scored data
# ------------------------------------------------------------------------------
 
score_df = spark.createDataFrame(score_labels_and_predictions)
score_df.createOrReplaceTempView("scoring_file")
spark.sql("drop table if exists my_db.pyspark_scored_sample")
spark.sql("create table my_db.pyspark_scored_sample (Customer_ID bigint, prediction int)")
spark.sql("insert into my_db.pyspark_scored_sample select * from scoring_file")
