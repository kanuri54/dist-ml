"""
DistML
=====
Provides

    1. Methods to distribute Machine Learning model training & inference.
    2. Tools like Tensorboard & HParams to visualize model metrics, graphs and hyper-parameter optimizations.
    3. Control of the entire processes through one YAML config file.
    
Available subpackages
---------------------
config
    Configuration class used to read config from YAML file.
data setup
    Method to read data from TFRecord files and make it available for executor consumption.
dml
    Core script which triggers the actions based on the subcommand.
train_and_eval
    Method to distribute training accross the grid. Currently supports Keras model.fit functionality alone.
inference
    Method to distribute inference accross the grid.
hparam_tuning
    Method to implement Hyper parameter optimization in distributed fashion.
utils
    Subpackage with helper methods 
    Refer Utiities section.

Utilities
---------
dml_helper
    Contains methods to delete nafs directories and to parse lin options.
spark_utils
    Consists method to create Spark instace with the setting from YAML file.
tbTool
    Generates Tensorboard instance on a random free port.
    Tensorboard can be used to visualize metrics.
    During hyper paramater tuning tensarboard displays HPARAMS dashboard.
timer
    Method used to time any function. Can be used as a decorator.
"""
