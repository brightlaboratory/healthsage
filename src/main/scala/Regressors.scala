import SimpleApp.toDouble
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.{GeneralizedLinearRegression, RandomForestRegressor}
import org.apache.spark.sql.DataFrame

object Regressors {
  //Regression
  def predictAverageTotalPaymentsUsingRandomForestRegression(origDf: DataFrame) = {

    // We will use AverageTotalPayments as the label

    val df = origDf.withColumn("label", origDf("AverageTotalPayments"))
      .withColumn("ProviderZipCodeDouble", toDouble(origDf("ProviderZipCode"))).withColumn("MedianHousePrice", toDouble(origDf("2011-12")))

    val feature1Indexer = new StringIndexer().setInputCol("DRGDefinition")
      .setOutputCol("feature1")
    val df_feature1 = feature1Indexer.fit(df).transform(df)

    val assembler = new VectorAssembler().setInputCols(Array("feature1",
      "ProviderZipCodeDouble", "MedianHousePrice")).setOutputCol("features")
    val df2 = assembler.transform(df_feature1)


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
    println("Random Forest Regresser model: " + model.toDebugString) //?
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

  // family can be: gaussian, Gamma
  def applyGeneralizedLinearRegression(origDf: DataFrame, family: String = "gaussian") = {
    // We will use AverageTotalPayments as the label

    val df = origDf.withColumn("label", origDf("AverageTotalPayments"))
      .withColumn("ProviderZipCodeDouble", toDouble(origDf("ProviderZipCode"))).withColumn("MedianHousePrice", toDouble(origDf("2011-12")))

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
