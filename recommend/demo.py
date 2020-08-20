from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark import SparkContext as sc
from pyspark import SparkConf
import numpy as np
import pandas as pd


df = pd.read_csv('../ratings.csv')

df = df.sample(frac=1.0)
train = df.loc[0:int(len(df)*0.8)]
test = df.loc[int(len(df)*0.8)+1:int(len(df))-1]

conf=SparkConf().setAppName("miniProject").setMaster("local[*]")
sc=sc.getOrCreate(conf)

als = ALS(maxIter=5, regParam=0.01, implicitPrefs=True,
          userCol="userId", itemCol="movieId", ratingCol="rating")
model = als.fit(train)

# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))