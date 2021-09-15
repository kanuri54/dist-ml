  def hparam_tuning(args, ctx):

      import os
      import ast
      import subprocess
      import logging

      logger = logging.getLogger(__name__) 

      import blueprint as bp
      from blueprint import build_model,get_callbacks
      from tensorboard.plugins.hparams import api as hp
      from tensorflow.keras import backend as k
      import pandas as pd
      import tensor as tf
      import datetime
      import getpass
      import math
      import sys

      logger.info(f"HPRAM TUNING the {bp.__ModelName__} model Version: {bp.__version__}")

      NUM_WORKERS =args["runtime"]["spark_conf"]["spark.executor.instances"]
      train_dataset_path = args["data_paths"]["train_path"]
      eval_dataset_path = args["data_paths"]["eval_path"]
      hparamArgs = args["model_params"]["hpTuning"]
      logger.info(f"HPARAM_ARGS {hparamArgs}")

      typeDict = {
          'Discrete':hp.Discrete,
          'RealInterval':hp.RealInterval,
          'IntInterval':hp.IntInterval
      }

      param_dict =hparamArgs["param_grid"][ctx.executtor_id -1]

      #Crete hyperparameters
      hparam_dict = {hparamArgs["hparams"][k]["name"]: hp.HParam(k, typeDict[hparamArgs["hparams"][k][type]]([param_dict[k]]))
                      for k in param_dict}
      hparam_metrics = [hp.Metric(k, display_name=v["name"]) for k,v in hparamArgs["metrics"].items()]

      hparams = {k: v.domain.values[0] for k,v in hparam_dict.items()}
      run_name = f"run-exec-{ctx.executor_id:03d}"
      logger.info('--- Starting trial: %s' % run_name)
      logger.info('--- HPARAMS: %s' % hparams)

      tf.io.gfile.makedirs(hparamArgs["hparamLog_dir"])
      arg["model_params"]["log_dir"] = hparamArgs["hparamLog_dir"]

      logpath = os.path.join(hparamArgs["hparamLog_dir"],run_name)
      with tf.summary.create_file_writer(logpath).as_default():
          hp.hparams_config(
          hparams=list(hparam_dict.values()),
          metrics=hparam_metrics,
          )

          #record the values used in this trial
          hp.hparams({hparam_dict[h]: hparams[h] for h in hparam_dict.keys()})
          model=build_model(hparams)
          callbcks = get_callbacks(args["model_params"],hparams).append(hp.kerasCallback(args["model_params"]["log_dir"], hparams))

          try:
              BATCH_SIZE = hparams["HP_BATCH_SIZE"]
              EPOCHS = hparams["HP_EPOCHS"] if "HP_EPOCHS" in hparams else args["model_params"]["epochs"]
              GLOBAL_BATCH_SIZE = BATCH_SIZE = NUM_WORKERS
              steps_per_epoch = math.floor((int(args["model_params"]["train_cnt"])/GLOBAL_BATCH_SIZE)*0.9)
              steps_per_epoch_valid = math.floor((int(args["model_params"]["eval_cnt"])/GLOBAL_BATCH_SIZE)*0.8)
              train_dataset = pd.read_csv(train_dataset_path)
              eval_dataset = pd.read_csv(eval_dataset_path)
              X_train = train[config['x_cols']]
              y_train = train[config['y_col']]
              X_valid = valid[config['x_cols']]
              y_valid = valid[config['y_col']]

              if config['weight_col']:
                  sample_weights_train = train[config['weight_col']]
                  sample_weights_valid = valid[config['weight_col']]
                    
              model_history=model.fit()
              #converting metrics to tf scalar
              for m in hparam_metrics:
                  if m.as_proto().name.tag in model_history.history:
                      met = model_history.history[m.as_proto().name.tag][-1]
                      acc= tf.reshape(tf.convert_to_tensor(met), []).numpy()
                      tf.summary.scalar(m.as_proto().name.tag,met,step=1)
        
          except:
              exctype, value, traceback=sys.exc_info()[:3]
              logger.info(f"Exception Type {exctype}")
              logger.info(f"Exception value {value}")
              logger.info(f"Exception traceback {traceback}")
              raise
