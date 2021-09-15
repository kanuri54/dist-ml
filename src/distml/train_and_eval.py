def train_and_eval(args, ctx):
    """This code runs on every single executor in the cluster.
    Do not attempt to read/write local files here. I/O needs to be through HDFS.
    """
    import os
    import subprocess
    import logging

    logger = logging.getLogger(__name__)

    import blueprint as bp
    from blueprint import build_model, get_calbacks, get_model_config
    from distml.data_setup import dataset_fn
    from tensorflow_estimator.python.estimator.export import export_lib
    from tensorflowonspark import compat
    import tensorflow as tf
    import pandas as pd
    import numpy as np
    import random
    import math
    import sys

    logger.info(f"training the {bp.__ModelName__} model Version: {bp.__version__}")

    # Set seeds for consistent results
    SEED = 2021
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(SEED)
    random.seed(seed)
    np.random.seed(seed)
    
    # control threading, which helps training speed
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    # instantiate distribution strategy
    strategy = tf.distribute.experimental.MultiWorkerMirrorStrategy()

    GLOBAL_BTCH_SIZE = args["model_params"]["batch_size"] = args["runtime"]["spark_conf"]["spark.executor.instances"]
    train_dataset_path =  args["data_paths"]["train_path"]
    eval_dataset_path =  args["data_paths"]["eval_path"]

    # generate train and eval datasets
    train_dataset = dataset_fn(train_dataset_path, input_context=ctx, args=args["model_params"])
    eval_dataset = dataset_fn(eval_dataset_path, input_context=ctx, args=args["model_params"])

    tf.io.gfile.makedirs(args["data_paths"]["export_dir"])
    export_dir = export_lib.get_timestamped_export_dir(args["data_paths"]["export_dir"])

    # get list of callbacks
    callbacks = get_callbacks(args["model_params"])

    # Note: if your part files have an uneven number of records, you may see an "out of Range" exception
    # at less than the expected number of steps_per_epoch, because the excutor with least amount of records will finish first.

    steps_per_epoch = math.floor((int(rgs["model_params"]["train_cnt"])/(GLOBAL_BATCH_SIZE))*0.9)
    steps_per_epoch_valid = math.floor((int(rgs["model_params"]["train_cnt"])/(GLOBAL_BATCH_SIZE))*0.8)

    try:
        with strategy.scope():
            model.build_model()

        model_history=model.fit(x=train_dataset, epochs=args["model_params"]["epochs"], steps_per_epoch=steps_per_epoch,
                                validation_data=eval_dataset,validation_steps=steps_per_epoch_valid,
                                callbacks=callbacks)
    except:
        ex_type, value, traceback=sys.exc_info()[:3]
        logger.error(f"Exception Type {exctype}")
        logger.error(f"Exception value {value}")
        logger.error(f"Exception traceback {traceback}")
        raise

Logger.info("========== exporting saved =_model to {}".format(export_dir))
compat.export_saved_model(model, export_dir, ctx.job_name == 'chief')
