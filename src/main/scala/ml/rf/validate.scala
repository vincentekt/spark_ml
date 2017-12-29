package ml.rf

import ml.common.{args_parse, feature_var}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.regression.RandomForestRegressionModel
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.functions.{col, exp}

object validate {
  def main(args: Array[String]): Unit = {

    val app_args = args_parse(args)

    val sc = new SparkContext(new SparkConf())
    val sqlContext = new SQLContext(sc)
    val testData = sqlContext.read.parquet(app_args.get('ipDataPath).get.asInstanceOf[String])

    val assembler = new VectorAssembler().setInputCols(feature_var).setOutputCol("features")

    val model = PipelineModel.load(app_args.get('ipModelPath).get.asInstanceOf[String])

    val colNames = Seq("y", "features")
    val test_data_raw = assembler.transform(testData).select(colNames.head, colNames.tail: _*)

    val predictions = model.transform(test_data_raw)

    val evaluator = new RegressionEvaluator()
      .setLabelCol("y")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    val rmse = evaluator.evaluate(predictions.withColumn("y", exp(col("y")))
      .withColumn("prediction", exp(col("prediction"))))
//    val rmse = evaluator.evaluate(predictions)
    val rfModel = model.stages(0).asInstanceOf[RandomForestRegressionModel]

    println("Learned regression RF model:\n" + rfModel.toDebugString)
    println("Root Mean Squared Error (RMSE) on test data = " + rmse)
    println("Feature Importances: " + rfModel.featureImportances)

  }
}
