import SimpleApp.toDouble
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.{GBTRegressor, GeneralizedLinearRegression, LinearRegression, RandomForestRegressor}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{avg, max, min, udf}

object Regressors {
  //Regression
  def predictAverageTotalPaymentsUsingRandomForestRegression(origDf: DataFrame) = {
    val df = addNumberOfDRGsforProviderAsColumn(origDf)
      .withColumn("label", origDf("AverageTotalPayments"))
      .withColumn("ProviderZipCodeDouble", toDouble(origDf("ProviderZipCode")))
      .withColumn("TotalDischargesDouble", toDouble(origDf("TotalDischarges")) )
      .withColumn("MedianHousePrice", toDouble(origDf("2011-12")))

    val feature1Indexer = new StringIndexer().setInputCol("DRGDefinition")
      .setOutputCol("feature1")
    val df_feature1 = feature1Indexer.fit(df).transform(df)

    //    val assembler = new VectorAssembler().setInputCols(Array(
    //      "ProviderZipCodeDouble", "TotalDischargesDouble", "MedianHousePrice")).setOutputCol("features")

    val assembler = new VectorAssembler().setInputCols(Array("feature1",
      "ProviderZipCodeDouble", "TotalDischargesDouble", "MedianHousePrice",
      "count(DISTINCT DRGDefinition)")).setOutputCol("features")

    val df2 = assembler.transform(df_feature1)

    val splitSeed = 5043
    val Array(trainingDataOrig, testDataOrig) = df2.randomSplit(Array(0.7, 0.3), splitSeed)

    val trainingData = trainingDataOrig.cache()
    val testData = testDataOrig.cache()

    import trainingData.sparkSession.implicits._
    val distinctDRGDefinitions = trainingData.select($"DRGDefinition").distinct()
      .rdd.map(row => row.getString(0)).collect()

    val predictions = applyRandomForestRegressionCore(trainingData, testData).cache()

    val overallAggError = computeAggregateError(computeError(predictions))
    val overallErrors = ("Overall", overallAggError._1, overallAggError._2, overallAggError._3,
      overallAggError._4, overallAggError._5, overallAggError._6,
      trainingData.count(), testData.count())

    // Here we will computer errors for each DRG separately
    val DRGErrors = distinctDRGDefinitions.sorted.map(DRG => {
      val subsetTrainingData = trainingData.where($"DRGDefinition".startsWith(DRG))
      val subsetTestData = testData.where($"DRGDefinition".startsWith(DRG))
      val subsetPredictions = predictions.where($"DRGDefinition".startsWith(DRG))
      val aggError =  computeAggregateError(computeError(subsetPredictions))
      (DRG, aggError._1, aggError._2, aggError._3, aggError._4, aggError._5, aggError._6,
        subsetTrainingData.count(), subsetTestData.count())
    }
    )

    // TODO: the outputFile must be different for each run or the previous output must be deleted.
    val outputFile = "wholeDRGError_Config1"
    testData.sparkSession.sparkContext.parallelize(Array(overallErrors) ++ DRGErrors)
      .toDF("DRG", "MinError", "AvgError", "MaxError", "MinPercentError", "AvgPercentError",
        "MaxPercentError", "TrainRows", "TestRows")
      .coalesce(1).write.option("header", "true").csv(outputFile)
  }

def predictAverageTotalPaymentsUsingGBT(origDf: DataFrame) = {

  // Using count(DISTINCT DRGDefinition) as a feature

  val df = addNumberOfDRGsforProviderAsColumn(origDf)
    .withColumn("label", origDf("AverageTotalPayments"))
    .withColumn("ProviderZipCodeDouble", toDouble(origDf("ProviderZipCode")))
    .withColumn("TotalDischargesDouble", toDouble(origDf("TotalDischarges")) )
    .withColumn("MedianHousePrice", toDouble(origDf("2011-12")))

  df.show()

  val feature1Indexer = new StringIndexer().setInputCol("DRGDefinition")
    .setOutputCol("feature1")
  val df_feature1 = feature1Indexer.fit(df).transform(df)

//  val assembler = new VectorAssembler().setInputCols(Array("feature1",
//    "ProviderZipCodeDouble", "MedianHousePrice","count(DISTINCT DRGDefinition)")).setOutputCol("features")

  val assembler = new VectorAssembler().setInputCols(Array("feature1",
    "ProviderZipCodeDouble", "TotalDischargesDouble", "MedianHousePrice",
    "count(DISTINCT DRGDefinition)")).setOutputCol("features")

  val df2 = assembler.transform(df_feature1)

  val splitSeed = 5043
  val Array(trainingData, testData) = df2.randomSplit(Array(0.7, 0.3), splitSeed)

  val gbt = new GBTRegressor()
    .setLabelCol("label")
    .setFeaturesCol("features")
    .setMaxIter(10)
    .setImpurity("variance")
    .setMaxDepth(3)
    .setMaxBins(1000)
    .setSeed(5043)

  val model = gbt.fit(trainingData)
  val predictions = model.transform(testData).cache()
  predictions.select("prediction", "label", "features").show(5)

  val evaluator = new RegressionEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")
    .setMetricName("rmse")
  val rmse = evaluator.evaluate(predictions)
  println("Root Mean Squared Error (RMSE) on test data = " + rmse)
  displayError(computeError(predictions))

}

  def applyRandomForestRegressionOnEachDRGSeparately(origDf: DataFrame) = {

    // We will use AverageTotalPayments as the label

    val df = addNumberOfDRGsforProviderAsColumn(origDf)
      .withColumn("label", origDf("AverageTotalPayments"))
      .withColumn("ProviderZipCodeDouble", toDouble(origDf("ProviderZipCode")))
      .withColumn("TotalDischargesDouble", toDouble(origDf("TotalDischarges")) )
      .withColumn("MedianHousePrice", toDouble(origDf("2011-12")))

//    val assembler = new VectorAssembler().setInputCols(Array(
//      "ProviderZipCodeDouble", "TotalDischargesDouble", "MedianHousePrice")).setOutputCol("features")

    val assembler = new VectorAssembler().setInputCols(Array(
      "ProviderZipCodeDouble", "TotalDischargesDouble", "MedianHousePrice",
      "count(DISTINCT DRGDefinition)")).setOutputCol("features")

    val df2 = assembler.transform(df)

    val splitSeed = 5043
    val Array(trainingDataOrig, testDataOrig) = df2.randomSplit(Array(0.7, 0.3), splitSeed)

    val traingData = trainingDataOrig.cache()
    val testData = testDataOrig.cache()

    import traingData.sparkSession.implicits._
    val distinctDRGDefinitions = traingData.select($"DRGDefinition").distinct()
      .rdd.map(row => row.getString(0)).collect()

    // Here we will build a model for each DRG separately
    val DRGErrors = distinctDRGDefinitions.sorted.zipWithIndex
      .filter(str_with_index => {
        if (str_with_index._2 >= 0 && str_with_index._2 <= 20) {
          true
        } else {
          false
        }
      })
      .map(DRG => {
        println("DRG: " + DRG)
        val subsetTrainingData = traingData.where($"DRGDefinition".startsWith(DRG._1))
        val subsetTestData = testData.where($"DRGDefinition".startsWith(DRG._1))
        val aggError = applyRandomForestRegressionCoreAndComputeErrors(subsetTrainingData, subsetTestData)
        (DRG._1, aggError._1, aggError._2, aggError._3, aggError._4, aggError._5, aggError._6,
          subsetTrainingData.count(), subsetTestData.count())
      }
      )

    // TODO: the outputFile must be different for each run or the previous output must be deleted.
    val outputFile = "PerDRGError_Config1_0_20"
    testData.sparkSession.sparkContext.parallelize(DRGErrors)
      .toDF("DRG", "MinError", "AvgError", "MaxError", "MinPercentError", "AvgPercentError",
        "MaxPercentError", "TrainRows", "TestRows")
      .coalesce(1).write.option("header", "true").csv(outputFile)

  }

  def applyRandomForestRegressionCore(trainingData: DataFrame, testData: DataFrame) = {
    // Random Forest Regresser
    val classifier = new RandomForestRegressor()
      .setImpurity("variance")
      .setMaxDepth(8)
      .setNumTrees(20)
      .setMaxBins(1000)
      .setFeatureSubsetStrategy("auto")
      .setSeed(5043)

    val model = classifier.fit(trainingData)
    //    println("Random Forest Regresser model: " + model.toDebugString)
    println("model.featureImportances: " + model.featureImportances)

    val predictions = model.transform(testData).cache()
    predictions.select("DRGDefinition", "TotalDischargesDouble", "count(DISTINCT DRGDefinition)","ProviderZipCode",
      "TotalDischarges", "MedianHousePrice", "features",
      "AverageTotalPayments", "label", "prediction").show(5, truncate = false)
    predictions
  }

  def applyRandomForestRegressionCoreAndComputeErrors(trainingData: DataFrame, testData: DataFrame) = {

    val predictions = applyRandomForestRegressionCore(trainingData, testData)
    val predictionWithErrors = computeAggregateError(computeError(predictions))
    predictionWithErrors
  }


  val difference = udf((label: Double, prediction: Double) => {
    Math.abs(label - prediction)
  })


  val percentDifference = udf((label: Double, prediction: Double) => {
    Math.abs(label - prediction) / Math.abs(label)
  })

  def computeError(df: DataFrame) = {
    df.withColumn("Difference", difference(df("label"), df("prediction")))
      .withColumn("PercentDifference", percentDifference(df("label"), df("prediction")))
  }

  def displayError(df: DataFrame) = {
    import df.sparkSession.implicits._
    println("The difference between label and prediction")
    df.select(min($"Difference"), avg($"Difference"), max($"Difference")).show(false)
    df.select(min($"PercentDifference"), avg($"PercentDifference"), max($"PercentDifference")).show(false)
  }

  def computeAggregateError(df: DataFrame) = {
    import df.sparkSession.implicits._
    val row1 = df.select(min($"Difference"), avg($"Difference"), max($"Difference")).first()
    val row2 = df.select(min($"PercentDifference"), avg($"PercentDifference"), max($"PercentDifference")).first()
    (row1.getDouble(0), row1.getDouble(1), row1.getDouble(2), row2.getDouble(0), row2.getDouble(1), row2.getDouble(2))
  }

  def addNumberOfDRGsforProviderAsColumn(df: DataFrame) = {

    df.createOrReplaceTempView("JoinedView")
    val groupedDf=df.sparkSession.sql("SELECT ProviderId,"+ "COUNT(DISTINCT DRGDefinition)" +
      "FROM JoinedView GROUP BY ProviderId ")

    val finalDF = df.join(
      groupedDf, df("ProviderId") === groupedDf("ProviderId"), "inner").cache()

    finalDF.show()
    finalDF.printSchema()

    finalDF

  }

  // family can be: gaussian, Gamma
  def applyGeneralizedLinearRegression(origDf: DataFrame, family: String = "gaussian") = {
    // We will use AverageTotalPayments as the label

    val df = origDf.withColumn("label", origDf("AverageTotalPayments"))
      .withColumn("ProviderZipCodeDouble", toDouble(origDf("ProviderZipCode"))).withColumn("MedianHousePrice", toDouble(origDf("2015-12")))

    val feature1Indexer = new StringIndexer().setInputCol("DRGDefinition")
      .setOutputCol("feature1")
    val df_feature1 = feature1Indexer.fit(df).transform(df)

    val assembler = new VectorAssembler().setInputCols(Array("feature1",
      "ProviderZipCodeDouble", "MedianHousePrice")).setOutputCol("features")
    val df2 = assembler.transform(df_feature1)


    val splitSeed = 5043
    val Array(trainingData, testData) = df2.randomSplit(Array(0.7, 0.3), splitSeed)

    // Generalized Linear Regression
    // Family: Gaussian

    val glr = new GeneralizedLinearRegression()
      .setFamily("gaussian")
      .setLink("identity")
      .setMaxIter(10)
      .setRegParam(0.3)

    // Fit the model on trainingData
    val model_glr = glr.fit(trainingData)


    // Apply the model on testData
    val predictions_glr = model_glr.transform(testData)
    predictions_glr.select("DRGDefinition", "ProviderZipCode", "MedianHousePrice",
      "AverageTotalPayments", "label", "prediction").show(5)

    val evaluator_glr = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("r2")

    val accuracy_glr = evaluator_glr.evaluate(predictions_glr)
    println("Accuracy GLR Gaussian Family: " + accuracy_glr)
  }

  //Linear Regression

  def applyLinearRegression(origDf: DataFrame) = {

    // Using count(DISTINCT DRGDefinition) as a feature

    val df = addNumberOfDRGsforProviderAsColumn(origDf)
      .withColumn("label", origDf("AverageTotalPayments"))
      .withColumn("count DRGDefinition", origDf("count(DISTINCT DRGDefinition)"))
      .withColumn("ProviderZipCodeDouble", toDouble(origDf("ProviderZipCode")))
      .withColumn("MedianHousePrice", toDouble(origDf("2011-12")))

    val feature1Indexer = new StringIndexer().setInputCol("DRGDefinition")
      .setOutputCol("feature1")
    val df_feature1 = feature1Indexer.fit(df).transform(df)

//    val assembler = new VectorAssembler().setInputCols(Array("feature1",
//      "ProviderZipCodeDouble", "MedianHousePrice","count(DISTINCT DRGDefinition)")).setOutputCol("features")

    val assembler = new VectorAssembler().setInputCols(Array("feature1",
      "ProviderZipCodeDouble", "TotalDischargesDouble", "MedianHousePrice",
      "count(DISTINCT DRGDefinition)")).setOutputCol("features")

    val df2 = assembler.transform(df_feature1)


    val splitSeed = 5043
    val Array(trainingData, testData) = df2.randomSplit(Array(0.7, 0.3), splitSeed)

    val lr = new LinearRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    // Fit the model
    val lrModel = lr.fit(trainingData)

    // Apply the model on testData
    val predictions_glr = lrModel.transform(testData)
    predictions_glr.select("DRGDefinition", "count DRGDefinition","ProviderZipCode", "MedianHousePrice",
      "AverageTotalPayments", "label", "prediction").show(5)

    val evaluator_glr = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("r2")

    val accuracy_lr = evaluator_glr.evaluate(predictions_glr)
    println("Accuracy Linear Regression: " + accuracy_lr)

  }
}
