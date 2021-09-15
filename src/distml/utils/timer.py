"""Timer utility. Allows on/off switch and directing logs to different targets""" 

import functools
import logging
import datetime

class Timer():
  """
  A Timer with context manger support.
  
  Usage
  1) as a decorator for functions, time a function call
  @Timer()
  def test():
    pass
  
  2) as a context manger, time a specific step
  with Timer():
      #do something here
    
  with Timer('specific step'):
    #do something here 

  Use Timer.print_time = false to turn off detailed timing logging.
  """
  print_time = True

  def __init__(self,process_name="", logger=logging.getLogger()):
      self.prosess_name = process_name
      self.logger = logger

  def __call__(self, func):
      @functools.wraps(func)
      def wrap(*args, **kwargs):
          if Timer.print_time:
              start = datetime.datetime.now()
          value = func(*args, **kwargs)
          if Timer.print_time:
            logger = logging.getLogger(func.__module)
            logger.info(
                f"{self.process_name if len(self.process_name) > 0 else func.__name__} elapsed time: {datetime.now() - start}")
            return value
      return wrap

def __enter__(self, ):
  if Timer.print_time:
        self.start = datetime.now()
  return self

def __exit__(self,*args):
    if Timer.print_time:
        selff.logger.info(
            f"{self.process_name + ' ' if len(self.process_name) > 0 else ''}elapsed time: {datetime.now() - selfstart}")
        
