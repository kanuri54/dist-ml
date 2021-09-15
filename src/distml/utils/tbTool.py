  import sys
  import os
  import logging
  from tensorboard import program
  import socket

  logger = logging.getLogger(__name__)

  def get_open_port():
      s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      s.bind(("",0))
      s.listen(1)
      port = s.getsockname()[1]
      s.close()
      return port

  class tensorboardSupervisor:
    def __init__(self, log_dir):
      self.tb_port = get_open_port()
      self.log_dir = log_dir
    
    def run(self):
      tb_url = f"http://{socket.gethostname()}.wellsfargo.com:{self.tb_port}"
      logger.info(f"\nTensorBoard 2.1.1 at {tb_url} (Press CTRL+C to quit)")
      os.sysytem(f'{sys.executable} -m tensorboard.main --logdir "{self.log_dir}"'
                f'--host=0.0.0.0 --port={self.tb_port} >/dev/null 2>&1')
      return tb_url
    
    def shutdown(self):
        logger.info('Killing Tensorboard Server')
        os.system(f'fuser -k {self.tb_port}/tcp')
