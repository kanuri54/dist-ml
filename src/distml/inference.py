import os, subprocess, logging
import tensorflow as tf
import numpy as np
import pandas as pd

def predict_udf(data, export_dir):
    from blueprint.modelUtils import get_model_config
    features, labels, weights, metadata = [], [], [], []
    for item in data:
      features.append(item[0])
      labels.append(item[1])
      weights.append(item[2])
      metadata.append(item[3])
    xs = np.array(features)
    ys = np.array(labels)
    ws = np.array(weights)
    ms = np.array(metadata)

    config = get_model_config()

    # Load the saved model and predict
    saved_model = tf.save_model.load(export_dir, tags='serve')
    predict = saved_model.signatures['serving_default']
    preds = predict(**{config['first_layer']:tf.convert_tensor(xs,np.float32)})
    return np.concatenate((xs, ys, preds[config['final_layer']].numpy(), ms, ws), axis=1)
	
def inference(args, ctx):
    from blueprint.modelUtils import get_model_config

    logger = logging.getLogger(__name__)

    pred_dataset_path = args["data_paths"]["pred_path"]
    output_dir = args["data_paths"]["output_dir"]
    config = get_model_config()
    tf.io.gfile.makedirs(output_dir)

    logger.info("Predicting using {bp.__ModelName__} Version: {bp.__version__}")

    # path to  saved_model
    model_path = os.path.join(args["data_paths"]["export_dir"], str(args["model_params"]["model_id"]))

    pred_dataset = spark.read.csv(pred_dataset_path)

    output_file = "{}/part-{:05d}.csv".format(output_dir, ctx.worker_num)

    out = pred_dataset.rdd\
          .map(lambda x: ([x[i] for i in config['x_cols']], [x[i] for i in config['y_cols']],
                  [x[i] for i in [config['weight_col']]], [x[i] for i in config['meta_data']])
            )\
          .mapPartitions(lambda rows: predict_udf(rows, model_path))
    df = out.map(lambda x: x.tolist()).toDF(config['x_cols'] + config['y_cols'] + [config['y_col']+'_predicted'] + config['meta_data'] + config['weight_col'])
    output_dir = args["data_paths"]["output_dir"]
    df.write.mode("overwrite").format("csv").save(output_file)
