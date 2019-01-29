# ------------------------------------------------------------------------------
# Importing required libraries
# ------------------------------------------------------------------------------
from pyspark.sql.types import Row

# Importing required libraries for VIF Calculation
from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import DenseVector
from pyspark.ml.linalg import Vectors
from pyspark.ml.evaluation import RegressionEvaluator

# ------------------------------------------------------------------------------
# Creating an Pyspark dataframe from a hive table
# ------------------------------------------------------------------------------
basedata = spark.table("my_database.my_table")

# ------------------------------------------------------------------------------
# Calculating VIF
# Assigning the threshold for VIF in the first line
# This may be changed to any other value as per requirement
# ------------------------------------------------------------------------------
vif_threshold = 5 #Threshold for VIF

def vif_cal_iter(inputdata,vif_threshold):
  xvar_names = inputdata.columns
  global vif_max
  global colnum_max
  colnum_max = 10000 # Initialising with a fake value
  vif_max = vif_threshold + 1
  def vif_cal(inputdata, xvar_names, vif_max, colnum_max, vif_threshold):
    print("Dimension of table at this level")
    print("================================")
    print(inputdata.count(), len(inputdata.columns))
    print("List of X Variables")
    print("===================")
    print(xvar_names)
    vif_max = vif_threshold
    for i in range(2,len(xvar_names)):
      train_t = inputdata.rdd.map(lambda x: [Vectors.dense(x[2:i]+x[i+1:]), x[i]]).toDF(['features', 'label'])
      lr = LinearRegression(featuresCol = 'features', labelCol = 'label')
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
    vif_max, colnum_max = vif_cal(inputdata, xvar_names, vif_max, colnum_max, vif_threshold)
    if vif_max > vif_threshold:
        print("Start of If Block")
        inputdata = inputdata.drop(inputdata[colnum_max])
        xvar_names = inputdata.columns
        print("Dimension of table after this iteration")
        print("=======================================")
        print(inputdata.count(), len(inputdata.columns))
        print("List of X Variables remaining")
        print("=============================")
        print(xvar_names)
  else:
    return inputdata

train = vif_cal_iter(basedata,vif_threshold)
print(train.count(), len(train.columns))
