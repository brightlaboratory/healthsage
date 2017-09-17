import SimpleApp.toDouble
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{avg, max, min}

object ClusteringAlgorithm {
  // K- Means Clustering

  def applyKmeans(origDf: DataFrame) = {

    import origDf.sparkSession.implicits._
    // We want to predict AverageTotalPayments as a function of DRGDefinition, and ProviderZipCode

    origDf.show()
    val df = origDf.withColumn("ProviderZipCodeDouble", toDouble(origDf("ProviderZipCode")))
      .withColumn("MedianHousePrice", toDouble(origDf("2011-12")))

    val feature1Indexer = new StringIndexer().setInputCol("DRGDefinition")
      .setOutputCol("feature1")

    val df_feature1 = feature1Indexer.fit(df).transform(df)

    val assembler = new VectorAssembler().setInputCols(Array("feature1",
      "ProviderZipCodeDouble", "MedianHousePrice")).setOutputCol("features")
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
    import df4.sparkSession.implicits._
    df4.select(min(df4("AverageTotalPayments")),
      avg(df4("AverageTotalPayments")), max(df4("AverageTotalPayments"))).show(false)

    println("Cluster 0 AverageTotalPayments:")

    df4.filter(df4("prediction") === 0).select(min(df4("AverageTotalPayments")),
      avg(df4("AverageTotalPayments")), max(df4("AverageTotalPayments"))).show(false)

    println("Cluster 1 AverageTotalPayments:")
    df4.filter(df4("prediction") === 1).select(min(df4("AverageTotalPayments")),
      avg(df4("AverageTotalPayments")), max(df4("AverageTotalPayments"))).show(false)

    println("Cluster 0 ProviderZipCodeDouble:")

    df4.filter(df4("prediction") === 0).select(min(df4("ProviderZipCodeDouble")),
      avg(df4("ProviderZipCodeDouble")), max(df4("ProviderZipCodeDouble"))).show(false)

    println("Cluster 1 ProviderZipCodeDouble:")
    df4.filter(df4("prediction") === 1).select(min(df4("ProviderZipCodeDouble")),
      avg(df4("ProviderZipCodeDouble")), max(df4("ProviderZipCodeDouble"))).show(false)

    // Evaluate clustering by computing Within Set Sum of Squared Errors.
    val WSSSE = model.computeCost(trainingData)
    println(s"Within Set Sum of Squared Errors = $WSSSE")

    println("Cluster centers:")
    model.clusterCenters.foreach(println)
  }
}
