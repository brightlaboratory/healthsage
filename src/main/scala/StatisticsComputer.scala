import SimpleApp.toDouble
import org.apache.spark.sql.DataFrame
import org.apache.spark.mllib.stat.Statistics

object StatisticsComputer {

  def computeCorrelations(df: DataFrame, column1: String, column2: String) = {
    val seriesX = toDoubleRDD(df.select(column1))
    val seriesY = toDoubleRDD(df.select(column2))

    // compute the correlation using Pearson's method. Enter "spearman" for Spearman's method. If a
    // method is not specified, Pearson's method will be used by default.
    val correlation: Double = Statistics.corr(seriesX, seriesY, "spearman")
    println(s"Correlation is: $correlation")
  }

  def toDoubleRDD(df: DataFrame) = {
    df.rdd.map(row => row.getAs[Double](0))
  }

  def computeStatsOnPaymentData(df: DataFrame) = {
    computeCorrelations(preprocessPaymentData(df), "MedianHousePrice", "AverageTotalPayments")
  }

  def preprocessPaymentData(df: DataFrame) = {
    df.where(df("DRGDefinition").startsWith("871"))
      .withColumn("MedianHousePrice", toDouble(df("2015-12")))
      .withColumn("ProviderZipCodeDouble", toDouble(df("ProviderZipCode")))
  }
}
