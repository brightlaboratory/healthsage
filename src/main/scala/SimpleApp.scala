/* SimpleApp.scala */


import org.apache.spark.SparkContext
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

    val df = normalizeHeaders(orig_df)
    calculateStats(df)
//    df.take(10).foreach(row => println("ROW: " + row))
  }

  def normalizeHeaders(df: DataFrame) = {
    var newDf = df
    for(col <- df.columns){
      newDf = newDf.withColumnRenamed(col,col.replaceAll("\\s", "_"))
    }

    newDf
  }

  def calculateStats(df: DataFrame): Unit = {
    df.createOrReplaceTempView("payment")
    df.sparkSession.sql("SELECT Provider_Zip_Code, COUNT(*) as count " +
      "FROM payment GROUP BY Provider_Zip_Code").show()

    df.sparkSession.sql("SELECT DISTINCT DRG_Definition FROM payment").show(10000)
  }
}