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

    config = get_model_config()
    
    train_dataset_path =  args["data_paths"]["train_path"]
    eval_dataset_path =  args["data_paths"]["eval_path"]

    # generate train and eval datasets
    train_dataset = pd.read_csv(train_dataset_path)
    eval_dataset = pd.read_csv(eval_dataset_path)

    X_train = train[config['x_cols']]
    y_train = train[config['y_col']]
    X_valid = valid[config['x_cols']]
    y_valid = valid[config['y_col']]
    
    if config['weight_col']:
        sample_weights_train = train[config['weight_col']]
        sample_weights_valid = valid[config['weight_col']]
        
    tf.io.gfile.makedirs(args["data_paths"]["export_dir"])
    export_dir = export_lib.get_timestamped_export_dir(args["data_paths"]["export_dir"])

    # get list of callbacks
    callbacks = get_callbacks(args["model_params"])

    if ctx.num_workers == 1:
        model = build_model()
    else:
        # instantiate distribution strategy
        strategy = tf.distribute.experimental.MultiWorkerMirrorStrategy() 
        
        # model building and compilinig should happen within scope
        with strategy.scope():
            model.build_model()
        
    # Note: if your part files have an uneven number of records, you may see an "out of Range" exception
    # at less than the expected number of steps_per_epoch, because the excutor with least amount of records will finish first.

    try:
        model_history=model.fit(X_train, y_train, sample_weight = sample_weights_train,
                                validation_data = (X_valid, y_valid, sample_weight_valid),
                                epochs=args["model_params"]["epochs"], callbacks=callbacks, verbose=2)
        
    except:
        ex_type, value, traceback=sys.exc_info()[:3]
        logger.error(f"Exception Type {exctype}")
        logger.error(f"Exception value {value}")
        logger.error(f"Exception traceback {traceback}")
        raise

Logger.info("========== exporting saved =_model to {}".format(export_dir))
compat.export_saved_model(model, export_dir, ctx.job_name == 'chief')
