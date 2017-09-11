
import org.apache.spark.sql.{SQLContext, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{BeforeAndAfter, FlatSpec}

class SimpleAppSpecs extends FlatSpec with BeforeAndAfter {

  private val master = "local[2]"
  private val appName = "example-spark"

  private var sc: SparkContext = _
  private var sparkSession: SparkSession = _

  before {
    val conf = new SparkConf()
      .setMaster(master)
      .setAppName(appName)

    sc = new SparkContext(conf)
    sparkSession = SparkSession.builder.
      master(master)
      .appName("spark session example")

      .getOrCreate()
  }

  after {
    if (sc != null) {
      sc.stop()
    }
  }

  "This test" should "count words" in {
    SimpleApp.countWords(sc)
  }

  "csvToDf" should "construct DF" in {
    SimpleApp.csvToDf(sparkSession)
  }
}
