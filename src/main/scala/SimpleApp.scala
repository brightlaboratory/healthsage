/* SimpleApp.scala */

import Regressors._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}

object SimpleApp {

  def predictPrices(spark: SparkSession) = {

    //Reading Inpatient_prospective_Payment_2015
    val orig_df = spark.read
      .option("header", "true") //reading the headers
      .csv(getClass.getClassLoader.getResource("Inpatient_Prospective_Payment_System__IPPS__Provider_Summary_for_All_Diagnosis-Related_Groups__DRG__-_FY2015.csv").getPath)

    val dollarColumns = Array("AverageCoveredCharges", "AverageTotalPayments",
      "AverageMedicarePayments")
    val paymentDf = removeLeadingDollarSign(normalizeHeaders(orig_df), dollarColumns)

    //Reading medicare_payment_2011.csv

    val orig_df2 = spark.read
      .option("header", "true") //reading the headers
      .csv(getClass.getClassLoader.getResource("medicare_payment_2011.csv").getPath)

    val dollarColumns2 = Array("AverageCoveredCharges", "AverageTotalPayments",
      "AverageMedicarePayments")
    val paymentDf2 = removeLeadingDollarSign(normalizeHeaders(orig_df), dollarColumns2)

    //Reading Zip_MedianValuePerSqft_AllHomes.csv
    val priceDf = spark.read
      .option("header", "true") //reading the headers
      .csv(getClass.getClassLoader.getResource("Zip_MedianValuePerSqft_AllHomes.csv").getPath)

    import spark.implicits._

    val joinedDf2=joinOnZipCode(paymentDf2,priceDf.where($"2011-12" isNotNull)).
      select("DRGDefinition", "ProviderId", "ProviderZipCode", "TotalDischarges", "2011-12", "AverageTotalPayments")

    joinedDf2.show()


    //val joinedDf = joinOnZipCode(paymentDf, priceDf.where($"2015-12" isNotNull))
     // .select("DRGDefinition", "ProviderId", "ProviderZipCode", "TotalDischarges", "2015-12", "AverageTotalPayments")

    Regressors.addNumberOfDRGsforProviderAsColumn(joinedDf2)
    //applyMachineLearningAlgorithms(joinedDf)
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


  def applyMachineLearningAlgorithms(df: DataFrame): Unit = {
    //    ClusteringAlgorithm.applyKmeans(df)
    //    Classifiers.applyNaiveBayesClassifier(df)
/// Regressors.predictAverageTotalPaymentsUsingRandomForestRegression(df)
    Regressors.applyRandomForestRegressionOnEachDRGSeparately(df)
    //Regressors.applyLinearRegression(df)
//    StatisticsComputer.computeStatsOnPaymentData(df)
    //    Regressors.applyGeneralizedLinearRegression(df, "gaussian")
    //    Regressors.applyGeneralizedLinearRegression(df, "Gamma")
  }
}
