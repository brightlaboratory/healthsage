/* SimpleApp.scala */


import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession

object SimpleApp {

  def countWords(sc: SparkContext): Unit = {
    val pathToFiles = "/Users/newpc/work/research/healthsage/pom.xml"
    val files = sc.textFile(pathToFiles)
    println("Count: " + files.count())
  }

  def csvToDf(spark: SparkSession) = {
    val df = spark.read
      .option("header", "true") //reading the headers
      .csv(getClass.getClassLoader.getResource("medicare_payment_2011.csv").getPath)

    df.take(10).foreach(row => println("ROW: " + row))
  }
}