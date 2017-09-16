/* SimpleApp.scala */


import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.{NaiveBayes, RandomForestClassifier}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.{GeneralizedLinearRegression, RandomForestRegressor}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}

object SimpleApp {

  def countWords(sc: SparkContext): Unit = {
    val pathToFiles = "/Users/anujatike/Documents/sem3/RA/healthsage-master/pom.xml"
    val files = sc.textFile(pathToFiles)
    println("Count: " + files.count())
  }

  def csvToDf(spark: SparkSession) = {

    //Reading medicare_payment_2011.csv

    val orig_df = spark.read
      .option("header", "true") //reading the headers
      .csv(getClass.getClassLoader.getResource("medicare_payment_2011.csv").getPath)

    val dollarColumns = Array("AverageCoveredCharges", "AverageTotalPayments",
      "AverageMedicarePayments")
    val paymentDf = removeLeadingDollarSign(normalizeHeaders(orig_df), dollarColumns)
    //calculateStats(paymentDf) // calling all other methods from here

    //Reading Zip_MedianValuePerSqft_AllHomes.csv


    val priceDf = spark.read
      .option("header", "true") //reading the headers
      .csv(getClass.getClassLoader.getResource("Zip_MedianValuePerSqft_AllHomes.csv").getPath)
    //calculatesStats2(priceDf)

    import spark.implicits._
    joinOnZipCode(paymentDf,priceDf.where($"2011-12" isNotNull))


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

  // Joining dataframes based on zipcode
  def joinOnZipCode(df1: DataFrame, df2: DataFrame): Unit =
  {
    println("df1.count(): " + df1.count())
    val joinedDf = df1.join(
    df2, df1("ProviderZipCode") === df2("RegionName"), "inner")

    println("joinedDf.count(): " + joinedDf.count())
//    System.exit(0)
    //joinedDf.show()

    val finalDf=joinedDf.select("DRGDefinition", "ProviderZipCode", "AverageTotalPayments","2011-12")

    finalDf.show()

    //finalDf.printSchema()

    calculateStats(finalDf) // calling all other methods from here

  }


  val doubleToLabel = udf((money: Double) => {
    (money.toInt - (money.toInt % 100)).toString
  })


// Classification

  def predictAverageTotalPayments(origDf: DataFrame) = {
    // We want to predict AverageTotalPayments as a function of DRGDefinition, and ProviderZipCode


    // We will use AverageTotalPayments as the label

    val df = origDf.withColumn("paymentLabel", doubleToLabel(origDf("AverageTotalPayments")))

    val feature1Indexer = new StringIndexer().setInputCol("DRGDefinition").setOutputCol("feature1")

    val df_feature1 = feature1Indexer.fit(df).transform(df)

    val feature2Indexer = new StringIndexer().setInputCol("ProviderZipCode").setOutputCol("feature2")

    val df_feature2 = feature2Indexer.fit(df_feature1).transform(df_feature1)

    val feature3Indexer= new StringIndexer().setInputCol("2011-12").setOutputCol("feature3")

    val df_feature3 = feature3Indexer.fit(df_feature2).transform(df_feature2)

    val assembler = new VectorAssembler().setInputCols(Array("feature1",
      "feature2","feature3")).setOutputCol("features")

    val df2 = assembler.transform(df_feature3)

    val labelIndexer = new StringIndexer().setInputCol("paymentLabel").setOutputCol("label")

    val labelIndexerModel = labelIndexer.fit(df2)

    val df3 = labelIndexer.fit(df2).transform(df2)

    df3.show(10)
    df3.printSchema()

    val splitSeed = 5043
    val Array(trainingData, testData) = df3.randomSplit(Array(0.7, 0.3), splitSeed)



    //Random Forest Classifier

    val classifier = new RandomForestClassifier()
      .setImpurity("gini")
      .setMaxDepth(5)
      .setNumTrees(20)
      .setFeatureSubsetStrategy("auto")
      .setMaxBins(100000)
      .setSeed(5043)

    val model = classifier.fit(trainingData)
    println("Random Forest Classifier model: " + model.toDebugString)
    println("model.featureImportances: " + model.featureImportances)
    //model.trees.foreach(tree => println("TREE: " + tree.toDebugString))

    val predictions = model.transform(testData)
    predictions.select("DRGDefinition", "ProviderZipCode","2011-12",
      "AverageTotalPayments","label", "prediction").show(5)

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val accuracy = evaluator.evaluate(predictions)
    println("Random Forest Classifier Accuracy: " + accuracy)

    val converter = new IndexToString().setInputCol("prediction")
      .setOutputCol("originalValue")
      .setLabels(labelIndexerModel.labels)
    val df4 = converter.transform(predictions)

    //df4.select("DRGDefinition", "ProviderZipCode", "2011-12",
     // "AverageTotalPayments", "label", "prediction",
      //"originalValue").show(5)

    //Naive Bayes Classifier

    // Train a NaiveBayes model.
    val model_naiveBayes = new NaiveBayes()
      .fit(trainingData)

    // Select example rows to display.
    val predictions_naiveBayes = model_naiveBayes.transform(testData)
    predictions_naiveBayes.show()

    // Select (prediction, true label) and compute test error
    val evaluator_naiveBayes = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val accuracy_naiveBayes = evaluator_naiveBayes.evaluate(predictions_naiveBayes)
    println("Naive Bayes Classifier Accuracy = " + accuracy_naiveBayes)

    val converter_naiveBayes = new IndexToString().setInputCol("prediction")
      .setOutputCol("originalValue")
      .setLabels(labelIndexerModel.labels)

    val df5 = converter_naiveBayes.transform(predictions_naiveBayes)

    df5.show()

    //df5.select("DRGDefinition", "ProviderZipCode","2012-11",
      //"AverageTotalPayments", "label", "prediction",
      //"originalValue").show(5)

  }


  val toDouble = udf((str: String) => {
    str.toDouble
  })


  //Regression

  def predictAverageTotalPaymentsUsingRegression(origDf: DataFrame) = {

    // We will use AverageTotalPayments as the label

    val df = origDf.withColumn("label", origDf("AverageTotalPayments"))
      .withColumn("ProviderZipCodeDouble", toDouble(origDf("ProviderZipCode"))).withColumn("MedianHousePrice",toDouble(origDf("2011-12")))

    val feature1Indexer = new StringIndexer().setInputCol("DRGDefinition")
      .setOutputCol("feature1")
    val df_feature1 = feature1Indexer.fit(df).transform(df)

    val assembler = new VectorAssembler().setInputCols(Array("feature1",
      "ProviderZipCodeDouble","MedianHousePrice")).setOutputCol("features")
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
    predictions.select("DRGDefinition", "ProviderZipCode","MedianHousePrice",
      "AverageTotalPayments","label", "prediction").show(50)

    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("r2")
    val accuracy = evaluator.evaluate(predictions)
    println("Random Forest Regresser Accuracy: " + accuracy)


    // Generalized Linear Regression


    for (a <- 1 to  3 )
        {
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
          predictions_glr.select("DRGDefinition", "ProviderZipCode","MedianHousePrice",
          "AverageTotalPayments", "label", "prediction").show(5)

          val evaluator_glr = new RegressionEvaluator()
          .setLabelCol("label")
          .setPredictionCol("prediction")
          .setMetricName("r2")

          val accuracy_glr = evaluator_glr.evaluate(predictions_glr)
          println("GLR Family Gaussian Iteration No: " + a)
          println("Accuracy GLR Gaussian Family: " + accuracy_glr)

          //Family: Gamma

          val glr2 = new GeneralizedLinearRegression()
            .setFamily("Gamma")
            .setLink("identity")
            .setMaxIter(10)
            .setRegParam(0.3)

          // Fit the model on trainingData
          val model_glr2 = glr2.fit(trainingData)


          // Apply the model on testData
          val predictions_glr2 = model_glr2.transform(testData)
          predictions_glr2.select("DRGDefinition", "ProviderZipCode","MedianHousePrice",
            "AverageTotalPayments", "label", "prediction").show(5)

          val evaluator_glr2 = new RegressionEvaluator()
            .setLabelCol("label")
            .setPredictionCol("prediction")
            .setMetricName("r2")

          val accuracy_glr2 = evaluator_glr2.evaluate(predictions_glr)

          println("GLR Family Gamma Iteration No: " + a)
          println("GLR Accuracy Gamma Family: " + accuracy_glr2)
        }

  }


  // K- Means Clustering

  def applyKmeans(origDf: DataFrame) = {


    // We want to predict AverageTotalPayments as a function of DRGDefinition, and ProviderZipCode

    origDf.show()


    val df = origDf.withColumn("ProviderZipCodeDouble", toDouble(origDf("ProviderZipCode")))
      .withColumn("MedianHousePrice",toDouble(origDf("2011-12")))

    val feature1Indexer = new StringIndexer().setInputCol("DRGDefinition")
      .setOutputCol("feature1")

    val df_feature1 = feature1Indexer.fit(df).transform(df)

    val assembler = new VectorAssembler().setInputCols(Array("feature1",
      "ProviderZipCodeDouble","MedianHousePrice")).setOutputCol("features")
    val df2 = assembler.transform(df_feature1)

    val splitSeed = 5043
    val Array(trainingData, testData) = df2.randomSplit(Array(0.7, 0.3), splitSeed)


    val kmeans = new KMeans().setK(2).setSeed(1L)
    val model = kmeans.fit(trainingData)

    println("kmeans model: " + model)
    println("kmeans model.summary.clusterSizes: " + model.summary.clusterSizes.deep)

    import trainingData.sparkSession.implicits._

    val df4 = model.transform(testData)

    df4.show(10, truncate = false)
    df4.printSchema()

    println("Whole dataset stats:")
    df4.select(min($"AverageTotalPayments"),
      avg($"AverageTotalPayments"), max($"AverageTotalPayments")).show(false)

    println("Cluster 0 AverageTotalPayments:")

    df4.filter($"prediction" === 0).select(min($"AverageTotalPayments"),
      avg($"AverageTotalPayments"), max($"AverageTotalPayments")).show(false)

    println("Cluster 1 AverageTotalPayments:")
    df4.filter($"prediction" === 1).select(min($"AverageTotalPayments"),
      avg($"AverageTotalPayments"), max($"AverageTotalPayments")).show(false)

    println("Cluster 0 ProviderZipCodeDouble:")

    df4.filter($"prediction" === 0).select(min($"ProviderZipCodeDouble"),
      avg($"ProviderZipCodeDouble"), max($"ProviderZipCodeDouble")).show(false)

    println("Cluster 1 ProviderZipCodeDouble:")
    df4.filter($"prediction" === 1).select(min($"ProviderZipCodeDouble"),
      avg($"ProviderZipCodeDouble"), max($"ProviderZipCodeDouble")).show(false)

    // Evaluate clustering by computing Within Set Sum of Squared Errors.
    val WSSSE = model.computeCost(trainingData)
    println(s"Within Set Sum of Squared Errors = $WSSSE")

    println("Cluster centers:")
    model.clusterCenters.foreach(println)
  }


  // Function which calls other functions

  def calculateStats(df: DataFrame): Unit = {
    applyKmeans(df)
    predictAverageTotalPayments(df)
    predictAverageTotalPaymentsUsingRegression(df)

  }


}
