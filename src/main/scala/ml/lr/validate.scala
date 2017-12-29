package ml.lr

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.functions.{col, exp}

object validate {
  def main(args: Array[String]): Unit = {
    val sc = new SparkContext(new SparkConf())
    val sqlContext = new SQLContext(sc)
    val testData = sqlContext.read.parquet("tmp_vincent/trivago/testData")

    val feature_var = Array("locale", "day_of_week", "hour_of_day", "agent_id", "entry_page", "traffic_type",
      "session_duration", "countLength", "logLikelihood", "durPerPage", "logPosterior", "weekend", "am",
      "peakHours", "evening", "sleepHours", "durCo", "cocounts", "avgCocounts")

    val assembler = new VectorAssembler().setInputCols(feature_var).setOutputCol("features")

    val model = PipelineModel.load("tmp_vincent/trivago/model/lr_ml")

    val colNames = Seq("y", "features")
    val test_data_raw = assembler.transform(testData).select(colNames.head, colNames.tail: _*)

    val predictions = model.transform(test_data_raw)

    val evaluator = new RegressionEvaluator()
      .setLabelCol("y")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    val rmse = evaluator.evaluate(predictions.withColumn("y", exp(col("y")))
      .withColumn("prediction", exp(col("prediction"))))

//    val lrModel = model.stages(0).asInstanceOf[LinearRegressionModel]
//    val trainingSummary = lrModel.summary
//    println(s"numIterations: ${trainingSummary.totalIterations}")
//    println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
//    trainingSummary.residuals.show()
    println("Root Mean Squared Error (RMSE) on test data = " + rmse)
  }
}
