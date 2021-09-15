def train_and_eval(args, ctx):
    """This code runs on every single executor in the cluster.
    Do not attempt to read/write local files here. I/O needs to be through HDFS.
    """
    import os
    import subprocess
    import logging

    logger = logging.getLogger(__name__)

    import blueprint as bp
    from blueprint import build_model, get_callbacks, get_model_config
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
    seed = 2021
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
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
    train_dataset = train_dataset[config['x_cols']]
    eval_dataset = pd.read_csv(eval_dataset_path)
    eval_dataset = eval_dataset[config['x_cols']]
    
    X_train = train_dataset[config['x_cols']]
    y_train = train_dataset[config['y_col']]
    X_valid = eval_dataset[config['x_cols']]
    y_valid = eval_dataset[config['y_col']]
    
    if config['weight_col']:
        sample_weights_train = train[config['weight_col']]
        sample_weights_valid = valid[config['weight_col']]
        
    tf.io.gfile.makedirs(args["data_paths"]["export_dir"])
    export_dir = export_lib.get_timestamped_export_dir(args["data_paths"]["export_dir"])

    # get list of callbacks
    callbacks = get_callbacks(args["model_params"])

    if ctx['num_workers'] == 1:
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
        model_history=model.fit(np.asarray(train_dataset).astype('float32'),
                                validation_split = 0.2,
                                epochs=args["model_params"]["epochs"], callbacks=callbacks, verbose=2)
        logger.info("========== exporting saved model to {} ==========".format(export_dir))
        model.save(args["data_paths"]["export_dir"])
        return model_history
#        compat.export_saved_model(model, export_dir, ctx.job_name == 'chief')
        
    except:
        ex_type, value, traceback=sys.exc_info()[:3]
        logger.error(f"Exception Type {ex_type}")
        logger.error(f"Exception value {value}")
        logger.error(f"Exception traceback {traceback}")
        raise