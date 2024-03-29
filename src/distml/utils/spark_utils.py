import os
from pyspark import SparkConf
from pyspark.sql import SparkSession
from distml.utils.timer import Timer

@Timer()
def start_spark(configs):
    """Configure and start spark session.

    Parameters
    ----------
    configs: distml.configuration
    """
    conf = SparkConf()
    conf.setAppName(configs['runtime'].get("app_name", "distml"))
    conf.setMaster(configs['runtime'].get("master", "local"))
    spark_settings = list(configs['runtime']["spark_conf"].items())
    conf.setAll(spark_settings)

    spark = SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()
    return spark
