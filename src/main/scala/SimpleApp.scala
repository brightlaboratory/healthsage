/* SimpleApp.scala */


import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}

object SimpleApp {

  def countWords(sc: SparkContext): Unit = {
    val pathToFiles = "/Users/newpc/work/research/healthsage/pom.xml"
    val files = sc.textFile(pathToFiles)
    println("Count: " + files.count())
  }

  def csvToDf(spark: SparkSession) = {
    val orig_df = spark.read
      .option("header", "true") //reading the headers
      .csv(getClass.getClassLoader.getResource("medicare_payment_2011.csv").getPath)

    val dollarColumns = Array("AverageCoveredCharges", "AverageTotalPayments",
      "AverageMedicarePayments")
    val df = removeLeadingDollarSign(normalizeHeaders(orig_df), dollarColumns)
    calculateStats(df)
  }

  def removeLeadingDollarSign(df: DataFrame, columnNames: Array[String]) = {
    var newDf = df
    val suffix = "_new"
    columnNames.foreach(v => newDf = newDf.withColumn(v + suffix, removeDollarSign(newDf(v))))

    // We remove old columns
    columnNames.foreach(v => newDf = newDf.drop(v))
    columnNames.foreach(v => newDf = newDf.withColumnRenamed(v + suffix, v))
    newDf
  }

  val removeDollarSign = udf((money: String) => {
    money.replaceAll("\\$", "").toDouble
  })

  def normalizeHeaders(df: DataFrame) = {
    var newDf = df
    for(col <- df.columns){
      newDf = newDf.withColumnRenamed(col,col.replaceAll("\\s", ""))
    }

    newDf
  }

  val doubleToLabel = udf((money: Double) => {
    (money.toInt - (money.toInt % 100)).toString
  })


  def predictAverageTotalPayments(origDf: DataFrame) = {
    // We want to predict AverageTotalPayments as a function of DRGDefinition, and ProviderZipCode
    origDf.select("DRGDefinition", "ProviderZipCode", "AverageCoveredCharges",
      "AverageTotalPayments", "AverageMedicarePayments").take(10)
      .foreach(v => println("ROW: " + v))

    // We will use AverageTotalPayments as the label
    val df = origDf.withColumn("paymentLabel", doubleToLabel(origDf("AverageTotalPayments")))

    val feature1Indexer = new StringIndexer().setInputCol("DRGDefinition").setOutputCol("feature1")
    val df_feature1 = feature1Indexer.fit(df).transform(df)

    val feature2Indexer = new StringIndexer().setInputCol("ProviderZipCode")
      .setOutputCol("feature2")
    val df_feature2 = feature2Indexer.fit(df_feature1).transform(df_feature1)

    val assembler = new VectorAssembler().setInputCols(Array("feature1",
      "feature2")).setOutputCol("features")
    val df2 = assembler.transform(df_feature2)

    val labelIndexer = new StringIndexer().setInputCol("paymentLabel").setOutputCol("label")
    val labelIndexerModel = labelIndexer.fit(df2)
    val df3 = labelIndexer.fit(df2).transform(df2)

    df3.createOrReplaceTempView("df3")
    df_feature1.sparkSession.sql("SELECT COUNT(DISTINCT feature2) FROM df3").show(10000)

    df3.show(10)
    df3.printSchema()

    val splitSeed = 5043
    val Array(trainingData, testData) = df3.randomSplit(Array(0.7, 0.3), splitSeed)

    val classifier = new RandomForestClassifier()
      .setImpurity("gini")
      .setMaxDepth(5)
      .setNumTrees(20)
      .setFeatureSubsetStrategy("auto")
      .setMaxBins(100000)
      .setSeed(5043)
    val model = classifier.fit(trainingData)
    println("model: " + model.toDebugString)
    println("model.featureImportances: " + model.featureImportances)
    model.trees.foreach(tree => println("TREE: " + tree.toDebugString))

    val predictions = model.transform(testData)
    predictions.select("DRGDefinition", "ProviderZipCode", "AverageCoveredCharges",
      "AverageTotalPayments", "AverageMedicarePayments", "label", "prediction").show(5)

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println("accuracy: " + accuracy)

    val converter = new IndexToString().setInputCol("prediction")
      .setOutputCol("originalValue")
      .setLabels(labelIndexerModel.labels)
    val df4 = converter.transform(predictions)

    df4.select("DRGDefinition", "ProviderZipCode", "AverageCoveredCharges",
      "AverageTotalPayments", "AverageMedicarePayments", "label", "prediction",
      "originalValue").show(5)
  }

  val toDouble = udf((str: String) => {
    str.toDouble
  })

  def predictAverageTotalPaymentsUsingRegression(origDf: DataFrame) = {
    // We want to predict AverageTotalPayments as a function of DRGDefinition, and ProviderZipCode
    origDf.select("DRGDefinition", "ProviderZipCode", "AverageTotalPayments").take(10)
      .foreach(v => println("ROW: " + v))

    // We will use AverageTotalPayments as the label
    val df = origDf.withColumn("label", origDf("AverageTotalPayments"))
      .withColumn("ProviderZipCodeDouble", toDouble(origDf("ProviderZipCode")))

    val feature1Indexer = new StringIndexer().setInputCol("DRGDefinition")
      .setOutputCol("feature1")
    val df_feature1 = feature1Indexer.fit(df).transform(df)

    val assembler = new VectorAssembler().setInputCols(Array("feature1",
      "ProviderZipCodeDouble")).setOutputCol("features")
    val df2 = assembler.transform(df_feature1)

    val splitSeed = 5043
    val Array(trainingData, testData) = df2.randomSplit(Array(0.7, 0.3), splitSeed)

    val classifier = new RandomForestRegressor()
      .setImpurity("variance")
      .setMaxDepth(8)
      .setNumTrees(20)
      .setMaxBins(100)
      .setFeatureSubsetStrategy("auto")
      .setSeed(5043)

    val model = classifier.fit(trainingData)
    println("model: " + model.toDebugString)
    println("model.featureImportances: " + model.featureImportances)

    val predictions = model.transform(testData)
    predictions.select("DRGDefinition", "ProviderZipCode",
      "AverageTotalPayments", "label", "prediction").show(50)

    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("r2")
    val accuracy = evaluator.evaluate(predictions)
    println("r2: " + accuracy)
  }

  def calculateStats(df: DataFrame): Unit = {
    predictAverageTotalPaymentsUsingRegression(df)
  }
}