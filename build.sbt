lazy val root = (project in file("."))
  .settings(
    organization  := "mllib.rf",
    name := "train",
    version := "0.1",
    scalaVersion  := "2.11.6",
    resolvers += "Typesafe Repository" at "http://repo.typesafe.com/typesafe/releases/",
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-core" % "2.1.1",
      "org.apache.spark" %% "spark-sql" % "2.1.1",
      "org.apache.spark" %% "spark-mllib" % "2.1.1",
      "com.aerospike" % "aerospike-client" % "4.0.4",
      "com.typesafe" % "config" % "1.3.1",
      "joda-time" % "joda-time" % "2.9.9",
      "org.json4s" %% "json4s-native" % "3.2.10",
      "org.json4s" %% "json4s-jackson" % "3.2.10",
      "com.aerospike" % "aerospike-client" % "4.0.4",
      "com.typesafe.akka" %% "akka-actor" % "2.3.8"
    ),
    mainClass in assembly := Some("mllib.rf.train"),
    assemblyMergeStrategy in assembly := {
      case PathList("org", "apache", xs @ _*) => MergeStrategy.first
      case PathList("org", "aopalliance", xs @ _*) => MergeStrategy.first
      case PathList("javax", "inject", xs @ _*) => MergeStrategy.first
      case PathList("overview.html", xs @ _*) => MergeStrategy.first
      case x => {
        val oldStrategy = (assemblyMergeStrategy in assembly).value
        oldStrategy(x)
      }
    }
  )