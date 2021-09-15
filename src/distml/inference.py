  def inference(args, ctx):

      import os
      import subprocess
      import logging

      logger = logging.getLogger(__name__)

      import blueprint as bp
      from blueprint import pred_fn
      from distml.data_setup import dataset_fn
      import tensorflow as tf
      import pandas as pd
      import numpy as np

      pred_dataset_path = args["data_paths"]["pred_path"]
      out_dir = args["data_paths"]["output_dir"]
      tf.io.gfile.makedirs(output_dir)

      logger.info("Predicting using {bp.__ModelName__} Version: {bp.__version__}")

      # load saved_model
      model_path = os.path.join(args["data_paths"]["export_dir"], str(args["model_parms"]["model_id"]))
      saved_model = tf.saved_model.load(model_path, tags='serve')
      predict = saved_model.signatures['serving_default']

      pred_dataset = dataset_fn(pred_dataset_path, input_context=ctx, args=args["model_params"], repeat_op=false)

      output_file = "{}/part-{:05d}.parquet".format(output_dir, ctx.worker_num)

      df = pd.DataFrame()
      for i,batch in enumerate(pred_dataset):
          df =df.append(pred_fn(predict, batch))
      df.columns = df.colums.astype(str)
      df.to_parquet(output_file)
