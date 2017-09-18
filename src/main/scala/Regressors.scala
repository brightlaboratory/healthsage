import SimpleApp.toDouble
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.{GeneralizedLinearRegression, RandomForestRegressor}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{avg, max, min, udf}

object Regressors {
  //Regression
  def predictAverageTotalPaymentsUsingRandomForestRegression(origDf: DataFrame) = {

    // We will use AverageTotalPayments as the label

    val df = origDf.withColumn("label", origDf("AverageTotalPayments"))
      .withColumn("ProviderZipCodeDouble", toDouble(origDf("ProviderZipCode")))
      .withColumn("MedianHousePrice", toDouble(origDf("2015-12")))

    val feature1Indexer = new StringIndexer().setInputCol("DRGDefinition")
      .setOutputCol("feature1")
    val df_feature1 = feature1Indexer.fit(df).transform(df)

    val assembler = new VectorAssembler().setInputCols(Array("feature1",
      "ProviderZipCodeDouble", "MedianHousePrice")).setOutputCol("features")
    val df2 = assembler.transform(df_feature1)

//    df2.createOrReplaceTempView("data")
//    df2.sparkSession.sql("SELECT DRGDefinition, COUNT(*) FROM data GROUP BY DRGDefinition ORDER BY COUNT(*)")
//      .coalesce(1).write.option("header", "true").csv("sample_file.csv")


    val splitSeed = 5043
    val Array(trainingData, testData) = df2.randomSplit(Array(0.7, 0.3), splitSeed)

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

    val predictions = model.transform(testData)
    predictions.select("DRGDefinition", "ProviderZipCode", "MedianHousePrice",
      "AverageTotalPayments", "label", "prediction").show(50)

    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("r2")
    val accuracy = evaluator.evaluate(predictions)
    println("Random Forest Regresser Accuracy: " + accuracy)
  }

  def applyRandomForestRegressionOnEachDRGSeparately(origDf: DataFrame) = {

    // We will use AverageTotalPayments as the label

    val df = addNumberOfDRGsforProviderAsColumn(origDf)
      .where(origDf("DRGDefinition").startsWith("871"))
      .withColumn("label", origDf("AverageTotalPayments"))
      .withColumn("ProviderZipCodeDouble", toDouble(origDf("ProviderZipCode")))
      .withColumn("TotalDischargesDouble", toDouble(origDf("TotalDischarges")) )
      .withColumn("MedianHousePrice", toDouble(origDf("2015-12")))

    println("df.count(): " + df.count())
    df.printSchema()

    df.createOrReplaceTempView("data")
    df.sparkSession.sql("SELECT MIN(AverageTotalPayments), AVG(AverageTotalPayments), MAX(AverageTotalPayments) FROM data").show(false)

//    val assembler = new VectorAssembler().setInputCols(Array(
//      "ProviderZipCodeDouble", "TotalDischargesDouble", "MedianHousePrice")).setOutputCol("features")

    val assembler = new VectorAssembler().setInputCols(Array(
      "ProviderZipCodeDouble", "MedianHousePrice")).setOutputCol("features")

    val df2 = assembler.transform(df)

    val splitSeed = 5043
    val Array(trainingData, testData) = df2.randomSplit(Array(0.7, 0.3), splitSeed)

    // Random Forest Regresser
    val classifier = new RandomForestRegressor()
      .setImpurity("variance")
      .setMaxDepth(8)
      .setNumTrees(20)
      .setMaxBins(100)
      .setFeatureSubsetStrategy("auto")
      .setSeed(5043)

    val model = classifier.fit(trainingData)
    //    println("Random Forest Regresser model: " + model.toDebugString)
    println("model.featureImportances: " + model.featureImportances)

    val predictions = model.transform(testData)
    predictions.select("DRGDefinition", "ProviderZipCode", "TotalDischarges", "MedianHousePrice",
      "AverageTotalPayments", "label", "prediction").show(5, false)

    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("r2")
    val accuracy = evaluator.evaluate(predictions)
    println("Random Forest Regresser Accuracy: " + accuracy)

    val predictionWithErrors = computeError(predictions)
    displayError(predictionWithErrors)
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

  def addNumberOfDRGsforProviderAsColumn(df: DataFrame) = {
    // TODO: Add the number of types of DRGDefinition associated with the ProviderId as a column

    df
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
}
