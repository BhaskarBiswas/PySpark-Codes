# ------------------------------------------------------------------------------
# Step 0: Importing required libraries
# ------------------------------------------------------------------------------
from pyspark.sql.types import Row
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType

from pyspark.sql.functions import *
from pyspark.sql.window import *

# Importing required libraries for VIF Calculation
from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import DenseVector
from pyspark.ml.linalg import Vectors
from pyspark.ml.evaluation import RegressionEvaluator

# Importing required libraries for Logistic Regression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import LogisticRegressionModel

# Importing required libraries for Random Forest
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Importing required libraries for Decision Tree
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# ------------------------------------------------------------------------------
# Step 1: Creating an Pyspark dataframe from a hive table
# ------------------------------------------------------------------------------
modeldata = spark.table("mydb.modeling_data")
scoredata = spark.table("mydb.scoring_data")

train, test = guestdata.randomSplit([0.6, 0.4], seed=12345)

# ------------------------------------------------------------------------------
# Step 2: Calculating VIF (can be skipped if data contains too many columns)
# ------------------------------------------------------------------------------
def vif_cal1(inputdata,testdata):
  xvar_names = inputdata.columns
  global vif_max
  global colnum_max
  colnum_max = 1000
  vif_max = 6
  def vif_cal(inputdata, xvar_names, vif_max, colnum_max):
    vif_max = 5
    for i in range(2,len(xvar_names)):
      train_t = inputdata.rdd.map(lambda x: [Vectors.dense(x[2:i]+x[i+1:]), x[i]]).toDF(['features', 'label'])
      lr = LinearRegression(featuresCol = 'features', labelCol = 'label', regParam=0.1)
      lr_model = lr.fit(train_t)
      predictions = lr_model.transform(train_t)
      evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='label')
      r_sq=evaluator.evaluate(predictions, {evaluator.metricName: "r2"})
      vif=1/(1-r_sq)
      if vif_max < vif:
        vif_max = vif
        colnum_max = i
    return vif_max, colnum_max
  while vif_max > 5:
    vif_max, colnum_max = vif_cal(inputdata, xvar_names, vif_max, colnum_max)
    if vif_max > 5:
        inputdata = inputdata.drop(inputdata[colnum_max])
        testdata = testdata.drop(testdata[colnum_max])
        xvar_names = inputdata.columns
  else:
    return inputdata, testdata

train, test = vif_cal1(train,test)

# ------------------------------------------------------------------------------
# Step 3a: Data conversion for Feature Importance via RandomForestClassifier
# ------------------------------------------------------------------------------
transformed_train_df = train.rdd.map(lambda x: [Vectors.dense(x[2:]), x[0]]).toDF(['features', 'label'])
#transformed_test_df = test.rdd.map(lambda x: [Vectors.dense(x[2:]), x[0]]).toDF(['features', 'label'])
#transformed_gd_df = guestdata.rdd.map(lambda x: [Vectors.dense(x[2:]), x[0]]).toDF(['features', 'label'])

labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(transformed_train_df)
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(transformed_train_df)

# ------------------------------------------------------------------------------
# Step 3b: Feature Importance from Random Forest
# ------------------------------------------------------------------------------
rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label', numTrees=10, maxDepth=5, minInstancesPerNode=500, seed=12345)
rf_Model = rf.fit(transformed_train_df)
Feature_Imp = rf_Model.featureImportances.toArray().tolist()
Feature_List = train.columns[2:]
t = zip(Feature_List, Feature_Imp)
schema = StructType([StructField("Feature", StringType(), True), StructField("Importance", FloatType(), True)])
VarImp = sqlContext.createDataFrame(t,schema).sort(desc("Importance"))
VarImp.show(100)

CumVarImp = VarImp.withColumn("CumImpTotal", sum(VarImp.Importance).over(Window.orderBy(VarImp.Importance.desc())))
TopVarDF = CumVarImp[CumVarImp.CumImpTotal <= 0.9]
top_var=TopVarDF.select('Feature').toPandas().Feature[:].tolist()

varlist=['y_var','customer_index']+top_var
scorevarlist=['customer_index']+top_var
topvar_train = train.select(varlist)
topvar_test = test.select(varlist)
topvar_score = scoredata.select(scorevarlist)
topvar_train.show(10)

# ------------------------------------------------------------------------------
# Step 4a: Data conversion for modeling
# ------------------------------------------------------------------------------
transformed_topvar_train_df = topvar_train.rdd.map(lambda x: [Vectors.dense(x[2:]), x[0]]).toDF(['features', 'label'])
transformed_topvar_test_df = topvar_test.rdd.map(lambda x: [Vectors.dense(x[2:]), x[0]]).toDF(['features', 'label'])
transformed_topvar_score_df = topvar_score.rdd.map(lambda x: [Vectors.dense(x[1:]), x[0]]).toDF(['features', 'customer_index'])

labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(transformed_topvar_train_df)
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(transformed_topvar_train_df)


# ------------------------------------------------------------------------------
# Step 4b: Logistic Regression model and accuracy
# ------------------------------------------------------------------------------
lr = LogisticRegression(labelCol="indexedLabel", featuresCol="indexedFeatures", fitIntercept=True, elasticNetParam=0.5)
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, lr])
lrModel = pipeline.fit(transformed_topvar_train_df)
predictions = lrModel.transform(transformed_topvar_test_df)

evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
lrAccuracy = evaluator.evaluate(predictions)
print("Accuracy = %g" % lrAccuracy)

# ------------------------------------------------------------------------------
# Step 4c: Decision Tree model and accuracy
# ------------------------------------------------------------------------------
dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", maxDepth=10, minInstancesPerNode=500)
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])
dtModel = pipeline.fit(transformed_topvar_train_df)
predictions = dtModel.transform(transformed_topvar_test_df)

evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
dtAccuracy = evaluator.evaluate(predictions)
print("Accuracy = %g" % dtAccuracy)

# ------------------------------------------------------------------------------
# Step 4d: Random Forest model and accuracy
# ------------------------------------------------------------------------------
rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=20, maxDepth=10, minInstancesPerNode=500)
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf])
rfModel = pipeline.fit(transformed_topvar_train_df)
predictions = rfModel.transform(transformed_topvar_test_df)

evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
rfAccuracy = evaluator.evaluate(predictions)
print("Accuracy = %g" % rfAccuracy)

# ------------------------------------------------------------------------------
# Step 4e: Gradient Boosted Tree model and accuracy
# ------------------------------------------------------------------------------
gbt = GBTClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", maxIter=20, maxDepth=10, minInstancesPerNode=500)
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, gbt])
gbtModel = pipeline.fit(transformed_topvar_train_df)
predictions = gbtModel.transform(transformed_topvar_test_df)

evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
gbtAccuracy = evaluator.evaluate(predictions)
print("Accuracy = %g" % gbtAccuracy)

# ------------------------------------------------------------------------------
# Step 5: Comparing Accuracy and Scoring with the best model
# ------------------------------------------------------------------------------
def compare_accuracy(lrAccuracy,dtAccuracy,rfAccuracy,gbtAccuracy,topvar_score):
    acc=[]
    acc.append(lrAccuracy)
    acc.append(dtAccuracy)
    acc.append(rfAccuracy)
    acc.append(gbtAccuracy)
    acclst=sorted(acc,reverse=True)
    max_accuracy = acclst[0]
    if (lrAccuracy == max_accuracy):
        predictions = lrModel.transform(transformed_topvar_score_df)
    elif (dtAccuracy == max_accuracy):
        predictions = dtModel.transform(transformed_topvar_score_df)
    elif (rfAccuracy == max_accuracy):
        predictions = rfModel.transform(transformed_topvar_score_df)
    elif (gbtAccuracy == max_accuracy):
        predictions = gbtModel.transform(transformed_topvar_score_df)
    final_hive = topvar_score.join(predictions, ["gst_ref_i"])
    new_final_table = final_hive.drop('features', 'indexedFeatures', 'rawPrediction', 'probability')
    new_final_table.show(10)
    return new_final_table

final_scored_table = compare_accuracy(lrAccuracy,dtAccuracy,rfAccuracy,gbtAccuracy,topvar_score)

# ------------------------------------------------------------------------------
# Step 6: Creating final scored data on hive location
# ------------------------------------------------------------------------------
final_scored_table.registerTempTable("temp_tbl")
spark.sql("drop table if exists mydb.scored_data")
spark.sql("create table mydb.scored_data as select * from temp_tbl")
