package mllib.rf

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.VectorAssembler

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.tree.model.RandomForestModel

object validate {
  def main(args: Array[String]): Unit = {
    val sc = new SparkContext(new SparkConf())
    val sqlContext = new SQLContext(sc)
    val testData = sqlContext.read.parquet("tmp_vincent/trivago/testData")

    val feature_var = Array("locale", "day_of_week", "hour_of_day", "agent_id", "entry_page", "traffic_type",
      "session_duration", "countLength", "logLikelihood", "durPerPage", "logPosterior", "weekend", "am",
      "peakHours", "evening", "sleepHours", "durCo", "cocounts", "avgCocounts")

    val assembler = new VectorAssembler().setInputCols(feature_var).setOutputCol("features")

    val model = RandomForestModel.load(sc, "tmp_vincent/trivago/model/rf_mllib")

    val colNames = Seq("y", "features")
    val test_data_raw = assembler.transform(testData).select(colNames.head, colNames.tail: _*)
    val test_data = MLUtils.convertVectorColumnsFromML(test_data_raw, "features").rdd
      .map(dp => new LabeledPoint(dp.getAs[Double]("y"), dp.getAs("features")))

    val labelAndPreds = test_data.map{ point =>
      val prediction = model.predict(point.features)
      (math.exp(point.label), math.exp(prediction))
    }

    val testMSE = labelAndPreds.map{ case(v, p) => math.pow((v - p), 2)}.mean()
    val testRMSE = math.pow(testMSE, 0.5)

    println("Test Root Mean Squared Error = " + testRMSE)
    println("Learned regression forest model:\n" + model.toDebugString)

  }
}
