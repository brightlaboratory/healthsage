import SimpleApp.doubleToLabel
import org.apache.spark.ml.classification.{NaiveBayes, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.sql.DataFrame

object Classifiers {

  def applyRandomForestClassifier(origDf: DataFrame) = {
    // We want to predict AverageTotalPayments as a function of DRGDefinition, and ProviderZipCode

    println("origDf.count(): " + origDf.count())

    // We will use AverageTotalPayments as the label

    val df = origDf.withColumn("paymentLabel", doubleToLabel(origDf("AverageTotalPayments")))

    val feature1Indexer = new StringIndexer().setInputCol("DRGDefinition").setOutputCol("feature1")

    val df_feature1 = feature1Indexer.fit(df).transform(df)

    val feature2Indexer = new StringIndexer().setInputCol("ProviderZipCode").setOutputCol("feature2")

    val df_feature2 = feature2Indexer.fit(df_feature1).transform(df_feature1)

    val feature3Indexer = new StringIndexer().setInputCol("2015-12").setOutputCol("feature3")

    val df_feature3 = feature3Indexer.fit(df_feature2).transform(df_feature2)

    val assembler = new VectorAssembler().setInputCols(Array("feature1",
      "feature2", "feature3")).setOutputCol("features")

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
      .setMaxBins(999)
      .setSeed(5043)

    println("classifier.getMaxBins: " + classifier.getMaxBins)
    val model = classifier.fit(trainingData)
    println("Random Forest Classifier model: " + model.toDebugString)
    println("model.featureImportances: " + model.featureImportances)
    //model.trees.foreach(tree => println("TREE: " + tree.toDebugString))

    val predictions = model.transform(testData)
    predictions.select("DRGDefinition", "ProviderZipCode", "2015-12",
      "AverageTotalPayments", "label", "prediction").show(5)

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

    df4.select("DRGDefinition", "ProviderZipCode", "2015-12",
      "AverageTotalPayments", "label", "prediction",
      "originalValue").show(5)
  }

  def applyNaiveBayesClassifier(origDf: DataFrame) = {
    val df = origDf.withColumn("paymentLabel", doubleToLabel(origDf("AverageTotalPayments")))

    val feature1Indexer = new StringIndexer().setInputCol("DRGDefinition").setOutputCol("feature1")

    val df_feature1 = feature1Indexer.fit(df).transform(df)

    val feature2Indexer = new StringIndexer().setInputCol("ProviderZipCode").setOutputCol("feature2")

    val df_feature2 = feature2Indexer.fit(df_feature1).transform(df_feature1)

    val feature3Indexer = new StringIndexer().setInputCol("2015-12").setOutputCol("feature3")

    val df_feature3 = feature3Indexer.fit(df_feature2).transform(df_feature2)

    val assembler = new VectorAssembler().setInputCols(Array("feature1",
      "feature2", "feature3")).setOutputCol("features")

    val df2 = assembler.transform(df_feature3)

    val labelIndexer = new StringIndexer().setInputCol("paymentLabel").setOutputCol("label")

    val labelIndexerModel = labelIndexer.fit(df2)

    val df3 = labelIndexer.fit(df2).transform(df2)

    df3.show(10)
    df3.printSchema()

    val splitSeed = 5043
    val Array(trainingData, testData) = df3.randomSplit(Array(0.7, 0.3), splitSeed)

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
  }
}
