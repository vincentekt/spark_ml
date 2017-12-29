package ml

import java.io.{BufferedWriter, File, FileWriter}

import ml.common.{args_parse, feature_var}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{SQLContext, Row}
import org.apache.spark.sql.functions.{col, exp}
import org.apache.spark.{SparkConf, SparkContext}

object predict {

  def write_scala(file_path: String, out_obj: Array[Row]): Unit = {
    // Writing output
    val file = new File(file_path)
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write("row_num" + ";" + "hits" + "\n")
    out_obj.foreach { x => bw.write(x.getAs[String]("mixid") + ";" + x.getAs[Double]("prediction").toString + "\n") }
    bw.close()
  }

  def main(args: Array[String]): Unit = {

    val app_args = args_parse(args)

    val sc = new SparkContext(new SparkConf())
    val sqlContext = new SQLContext(sc)

    val testData = sqlContext.read.parquet(app_args.get('ipDataPath).get.asInstanceOf[String])

    val assembler = new VectorAssembler().setInputCols(feature_var).setOutputCol("features")

    val model = PipelineModel.load(app_args.get('ipModelPath).get.asInstanceOf[String])

    val colNames = Seq("mixid", "features")
    val test_data_raw = assembler.transform(testData).select(colNames.head, colNames.tail: _*)

    val predictions = model.transform(test_data_raw)

    val opColNames = Seq("mixid", "prediction")
    predictions.take(5).foreach(println)
    val local_pred = predictions.withColumn("prediction", exp(col("prediction")))
      .select(opColNames.head, opColNames.tail: _*).collect()

    write_scala(app_args.get('opDataPath).get.asInstanceOf[String], local_pred)
  }
}
