  import argparse
  import logging
  import sys
  import traceback
  import tensorflow as tf
  from tensorflowonspark import TFCluster
  from sklearn.model_selection import ParameterGrid

  from distml.train_and_eval import train_and_eval
  from distml.inference import inference
  from distl.hparam_tuning import hparam_tuning
  from distml.config import Configuration
  from distml.utils.spark_utils import start_spark
  from distml.utils.dml_helper import run_cmd, parse_cmd_options, hdfsDelete
  from distml.utils.timer import Timer
  from distml.utils.tbTool import TensorboardSupervisor

  logger = logging.getLogger(__name__)

  @Timer()
  def dml(args=[]):

      """Engine entry point. Set up logging, parse yaml,run forcast engie.

      Parameters
      ----------
      args: list
           Command line options, currently supports -c file_path.yml
      """
      logger = logging.getLogger("distml")
      logger.propagate = False
      handler = logging.StreamHandler()
      formatter = logging.formatter("%(asctime)s | %(name)s %(levelname)s: %(message)s")
      handler.setFormatter(formatter)
      logger.addHandler(handler)
      logger.setLevel(logging.INFO)

      args = parse_cmd_options(args)

      try:
          cmd, args = args.func(args)

          logger.info(f"Running function ... {cmd}")

          args = configuration.from_file(args.config)

          if not args["runtime"].get('print_time', False):
              Timer.print_time = False

          if not cmd=="run_tensorboard":

             spark = start_spark(args)

             sc=spark.spark(args)

             for path in args["runtime"]["library_paths"]:
                 logger.info(f"Uploading zip file from {path}")
                 sc.adddPyFile(path)

             logger.info(f"Spark session started. Application Id: {sc.application}")
             logger.info(f"Spark UI: {sc.uiWebUrl}")

          if cmd=="train_and_eval":

             if args["model_parms"].get("train_cnt")==None or args["model_parms"].get("eval_cnt")==None:
                args["model_parms"]["train_cnt"]=spark.read.format("tfrecords").option("recordType","Example").load(args["data_paths"]["train_path"]).count()
                args["model_parms"]["eval_cnt"]=spark.read.format("tfrecords").option("recordType","Example").load(args["data_paths"]["eval_path"]).count()

             logger.info(f'Train Data count : {args["model_params"]["train_cnt"]}. Validation Data Count : {args["model_params"]["eval_cnt"]}')

             hdfsDelete(args["data_paths"]["export_dir"])
             hdfsDelete(args["model_params"]["log_dir"])

             logger.info("Starting to train the model on shared grid ...")
        
             cluster = TFCluster.run(sc,train_and_eval, args, args["runtime"]["spark_conf"]["spark.executor.instances"],num_ps=0,
                                     tensorboard=args["model_parms"]["tensorboard"], log_dir=args["model_parms"]["log_dir"],
                                    input_mode=TFCluster.InputMode.TENSORFLOW, master_node='chief')
             cluster.shutdown()

             logger.info("Completed training the model.")

          elif cmd=="inference":
             
             hdfsDelete(args["data_paths"]["output_dir"])

             # Running single-node TF instances on each executor
             # TFParallel.run(sc, inference, args, args.cluster_size)
             logger.info("Starting model inferencing ...")
             cluster = TFCluster.run(sc, inference, args, args["runtime"]["spark_conf"]["spark.executor.instances"],num_ps=0,
                                    input_mode=TFCluster.InputMode.TENSORFLOW, master_node='chief')
             
             cluster.shutdown()

             logger.info("Completed the predictions.")

          elif cmd=="hpram_tuning":
              
             hdfsDelete(args["model_params"]["log_dir"])

             # clear hparamsLog_dir folder
             hdfsDelete(args["model_params"]["hpTuning"]["hparamlog_dir"])

             param_grid = list(ParameterGrid({k:v["values"] for k,v in args["model_params"]["hpTuning"]["hparams"].items()}))
             args["model_params"]["hpTuning"]["hparams"] = param_grid
             # Running single-node Tf instances on each executor
             # TFparallel.run(sc, inference, args, args.cluster_size)
             logger.info("Starting hyper-parameter tuning the model ...")
             cluster = TFCluster.run(sc, hparams_tuning, args, args["runtime"]["spark_conf"]["spark.executor.instances"],num_ps=0,
                                     tensorboard=args["model_parms"]["tensorboard"], log_dir=args["model_parms"]["hparamslog_dir"],
                                    input_mode=TFCluster.InputMode.TENSORFLOW, master_node='chief')
             
             cluster.shutdown()

             logger.info("Completed tuning the model.")

          elif cmd=="run_tensorboard":
             try:
                 tbSupervisor = TensorboardSupervisor(args["model_params"]["log_dir"])
                 tb_url = tbSupervisor.run()

             except KeyboardInterrupt:
                 logger.info("Attempting to shutdown TensorboardSupervisor process...")
                 tbSupervisor.shutdown()
                 sys.exit()

      except:
            exctype, value, traceback=sys.exc_info()[:3]
            logger.error(f"Exception Type {exctype}")
            logger.error(f"Exception Value {value}")
            logger.error(f"Exception Traceback {traceback}")
            raise
      finally:
            if not cmd=="run_tensorboard":
                logger.info("Closing spark context...")
                spark.stop()
