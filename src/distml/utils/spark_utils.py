  import os
  from pyspark import SparkConf
  from pyspark.sql import sparkSession
  from distml.utis.timer import timer
  
  @Timer()
  def start_spark(configs):
      """Configure and start spark session.

      Parameters
      ----------
      configs: distml.configuration
      """
      conf = SparkConf()
      conf.setAppName(configs.runtime,get("app_name", "distml"))
      conf.setMaster(configs.runtime,get("master", "yarn"))
      sparj_settings = list(configs.runtime["spark_conf"].items())
      conf.setAll(spark_settings)
      conf.set('spark.executorEnv.CLASSPATH',os.environ["CLASSPATH"])

      spark = SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()
      return spark
