
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
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

    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)
  }

  after {
    if (sc != null) {
      sc.stop()
    }
  }


  "predictPrices" should "predict medical prices" in {
    SimpleApp.predictPrices(sparkSession)
  }
}
