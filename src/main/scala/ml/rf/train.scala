package ml.rf

import ml.common.{args_parse, feature_var}
import org.apache.spark.ml.Pipeline
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{GBTRegressor, RandomForestRegressionModel, RandomForestRegressor}

object train {
  def main(args: Array[String]): Unit = {

    val app_args = args_parse(args)

    val sc = new SparkContext(new SparkConf())
    val sqlContext = new SQLContext(sc)
    val trainingData = sqlContext.read.parquet(app_args.get('ipDataPath).get.asInstanceOf[String])

    val assembler = new VectorAssembler().setInputCols(feature_var).setOutputCol("features")

    val colNames = Seq("y", "features")
    val train_data_raw = assembler.transform(trainingData).select(colNames.head, colNames.tail: _*)
    //    val train_data = MLUtils.convertVectorColumnsFromML(train_data_raw, "features").rdd.map(dp => new LabeledPoint(dp.getAs[Double]("y"), dp.getAs("features")))

    val rf = new RandomForestRegressor()
      .setLabelCol("y")
      .setFeaturesCol("features")

    val pipeline = new Pipeline().setStages(Array(rf))

    val model = pipeline.fit(train_data_raw)

//    model.stages(0).asInstanceOf[RandomForestRegressionModel].featureImportances

    model.write.overwrite().save(app_args.get('opModelPath).get.asInstanceOf[String])
  }
}
