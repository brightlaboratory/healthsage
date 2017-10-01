/* SimpleApp.scala */

import Regressors.addNumberOfDRGsforProviderAsColumn
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}

object SimpleApp {

  def predictPrices(spark: SparkSession) = {

    //Reading Inpatient_prospective_Payment_2015
    val df_2015 = spark.read
      .option("header", "true") //reading the headers
      .csv(getClass.getClassLoader.getResource("Inpatient_Prospective_Payment_System__IPPS__Provider_Summary_for_All_Diagnosis-Related_Groups__DRG__-_FY2015.csv").getPath)

    val dollarColumns = Array("AverageCoveredCharges", "AverageTotalPayments",
      "AverageMedicarePayments")
    val paymentDf = removeLeadingDollarSign(normalizeHeaders(df_2015), dollarColumns)

    //Reading medicare_payment_2011.csv

    val df_2011 = spark.read
      .option("header", "true") //reading the headers
      .csv(getClass.getClassLoader.getResource("medicare_payment_2011.csv").getPath)

    val dollarColumns2 = Array("AverageCoveredCharges", "AverageTotalPayments",
      "AverageMedicarePayments")
    val paymentDf_2011 = removeLeadingDollarSign(normalizeHeaders(df_2011), dollarColumns2)

    //Reading Zip_MedianValuePerSqft_AllHomes.csv
    val priceDf = spark.read
      .option("header", "true") //reading the headers
      .csv(getClass.getClassLoader.getResource("Zip_MedianValuePerSqft_AllHomes.csv").getPath)

    import spark.implicits._

    val joinedDf_2011 =joinOnZipCode(paymentDf_2011,priceDf.where($"2011-12" isNotNull)).
      select("DRGDefinition", "ProviderId", "ProviderZipCode", "TotalDischarges", "2011-12", "AverageTotalPayments")

    joinedDf_2011.show()


    //val joinedDf = joinOnZipCode(paymentDf, priceDf.where($"2015-12" isNotNull))
     // .select("DRGDefinition", "ProviderId", "ProviderZipCode", "TotalDischarges", "2015-12", "AverageTotalPayments")


    applyMachineLearningAlgorithms(joinedDf_2011)
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
    for (col <- df.columns) {
      newDf = newDf.withColumnRenamed(col, col.replaceAll("\\s", ""))
    }

    newDf
  }

  // Joining dataframes based on zipcode
  def joinOnZipCode(df1: DataFrame, df2: DataFrame) = {
    val joinedDf = df1.join(
      df2, df1("ProviderZipCode") === df2("RegionName"), "inner")
    joinedDf
  }


  val doubleToLabel = udf((money: Double) => {
    (money.toInt - (money.toInt % 100)).toString
  })

  val toDouble = udf((str: String) => {
    str.toDouble
  })

  val LongtoDouble = udf((num: Long) => {
    num.toDouble
  })


  def applyMachineLearningAlgorithms(df: DataFrame): Unit = {
    //    ClusteringAlgorithm.applyKmeans(df)
    //    Classifiers.applyNaiveBayesClassifier(df)
//    Regressors.predictAverageTotalPaymentsUsingRandomForestRegression(df)

    //generateAdHocStats(df)
//    Regressors.predictAverageTotalPaymentsUsingGBT(df)
//    Regressors.applyRandomForestRegressionOnEachDRGSeparately(df)
    Regressors.applyLinearRegression(df)
//    StatisticsComputer.computeStatsOnPaymentData(df)
    //    Regressors.applyGeneralizedLinearRegression(df, "gaussian")
    //    Regressors.applyGeneralizedLinearRegression(df, "Gamma")

    //Regressors.addNumberOfDRGsforProviderAsColumn(df)
  }

  def generateAdHocStats(df: DataFrame) = {
//    df.withColumn("TotalDischargesDouble", toDouble(df("TotalDischarges")) )
//      .createOrReplaceTempView("data")
//    df.sparkSession.sql("SELECT SUM(TotalDischargesDouble * AverageTotalPayments) FROM data").show(false)

//    df.sample(withReplacement = false, 5.toDouble / df.count().toDouble).show(truncate = false)
//      df.createOrReplaceTempView("data")
//    df.sparkSession.sql("SELECT COUNT(*), MIN(AverageTotalPayments), AVG(AverageTotalPayments), MAX(AverageTotalPayments), stddev_pop(AverageTotalPayments), stddev_pop(AverageTotalPayments)* 100 /AVG(AverageTotalPayments) FROM data GROUP BY DRGDefinition ORDER BY COUNT(*)")
//      .coalesce(1).write.option("header", "true").csv("standard_deviation")
    import df.sparkSession.implicits._
//    df.withColumn("MedianHousePrice", toDouble($"2011-12"))
//      .select($"DRGDefinition", $"MedianHousePrice", $"AverageTotalPayments")
//      .where($"DRGDefinition".startsWith("194"))
//      .orderBy($"MedianHousePrice")
//      .coalesce(1).write.option("header", "true").csv("194_house_to_payment")

//    addNumberOfDRGsforProviderAsColumn(df)
//      .where($"DRGDefinition".startsWith("194"))
//      .orderBy($"count(DISTINCT DRGDefinition)")
//      .select($"DRGDefinition", $"count(DISTINCT DRGDefinition)", $"AverageTotalPayments")
//      .coalesce(1).write.option("header", "true").csv("194_DRGCount_Payment")

//    StatisticsComputer.computeCorrelations(addNumberOfDRGsforProviderAsColumn(df)
//      .withColumn("MedianHousePrice", toDouble($"2011-12"))
//      .withColumn("DistinctDRGCount", LongtoDouble($"count(DISTINCT DRGDefinition)"))
//      .where($"DRGDefinition".startsWith("194")),
//      "DistinctDRGCount", "AverageTotalPayments", "pearson")

    //df.createOrReplaceTempView("data")
//    df.sparkSession.sql("SELECT COUNT(*) FROM data").show()
    //df.sparkSession.sql("SELECT * FROM data WHERE ProviderZipCode IN (SELECT ProviderZipCode FROM data GROUP BY ProviderZipCode HAVING COUNT(DISTINCT ProviderId) > 1) ORDER BY ProviderZipCode").show(1000, truncate = false)

  }
}
