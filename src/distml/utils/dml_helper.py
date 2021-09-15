import os
import logging
import argparse
import subprocess
from typing import List
from pyarrow import hdfs

logger = logging.getLogger("__name__")

def run_cmd(cmd):
    """
    run linux commands
    """
    logger.info(f'Running system comand: {cmd}')
    proc = subprocess.open(cmd, stdout=subprocess.PIPE, shell=True)
    s_output, s_err= proc.communicate()
    s_return = proc.returncode
    return s_return, s_output, s_err

def hdfsClearDir(path):
    try:
        fs = hdfs.connect(os.environ['HDFS_NAMESPACE'])
        if fs.exists(path):
            logger.info(f"Atemping to clear items in the path : {path}")
            for i in fs.ls(path):
                fs.delete(i, recursive=True) 
    except Excepation as e:
        logger.error(f"Exception occured: {e}")
    finally:
        fs.close()

def hdfsDelete(path):
    try:
        fs = hdfs.connect(os.environ['HDFS_NAMESPACE'])
        if fs.exists(path):
            logger.info(f"Attempting to delete the path : {path}")
            fs.delete(path, recursive=True)
    except Exception as e:
        logger.info(f"Exception occured:{e}")
    finally:
        fs.close()
def runTensorboard(args):
    return "run_tensorboard", args
def hparamTuning(args):
    return "hparam_tuning", args
def distTrainEval(args):
    return "train_and_eval", args 
def distPredict(args):
    return "inference", args
def parse_cmd_options(args: List[str]):
    """parse out command line options.
    parameters:
    args: list
      options give on the command line
    Returns
    -------
    map
      keys-value paris of the command line options
    """
    parser = argparse.ArgumentParser()
    
    subparsers = parser.add_subparsers(
        help='commands'
    )
    trainEvalparser = subparsers.add_parser('train_and_eval', help='train and eval in dstributed fashion')
    trainEvalparser.add_argument("-c","--config", required=True, help="configuration file loction")
    trainEvalparser.set_defalts(func=distTrainEval)
    predparser = subparsers.add_parser('perdict', help='predict in distributed fashion')
    predparser.add_argument(
        "-c", "--config",required=True,help ="condiguration file location"
    )
    predParser.set_defalts(func=distPredict)
    hpparser = subparsers.add_parser('hpTuning', help='run Hyper parameters tuning & view in tb dashboard ')
    hpparser.add_argument(
        "-c", "--config",required=True,help ="condiguration file location"
    )
    hpparser.set_defalts(func=hparamTuning)
    
    tbParser = subparsers.add_parser('tb', help='run tensorboard instance to visualize metrics or hparams')
      
    tbParser.add_argument(
        "-c", "--config",required=True,help ="condiguration file location"
    )
    
    tbParser.setdefaults(func=runTensorboard)
    
    return parser.parse_args(args)