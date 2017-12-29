package mllib.rf

import ml.common.{args_parse, feature_var}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.util.MLUtils

//import org.apache.spark.sql.functions.{isnull, when, count, col}
//trainingData.select(trainingData.columns.map(c => sum(col(c).isNull.cast("int")).alias(c)): _*).show()

object train {
  def main(args: Array[String]): Unit = {

    val app_args = args_parse(args)

    val sc = new SparkContext(new SparkConf())
    val sqlContext = new SQLContext(sc)
    val trainingData = sqlContext.read.parquet(app_args.get('ipDataPath).get.asInstanceOf[String])

    val assembler = new VectorAssembler().setInputCols(feature_var).setOutputCol("features")

    val colNames = Seq("y", "features")
    val train_data_raw = assembler.transform(trainingData).select(colNames.head, colNames.tail: _*)
    val train_data = MLUtils.convertVectorColumnsFromML(train_data_raw, "features").rdd.map(dp => new LabeledPoint(dp.getAs[Double]("y"), dp.getAs("features")))

    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 50 // Use more in practice.
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    val impurity = "variance"
    val maxDepth = 10
    val maxBins = 32

    val model = RandomForest.trainRegressor(train_data, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

    model.save(sc, "tmp_vincent/trivago/model/rf_mllib")

  }
}
