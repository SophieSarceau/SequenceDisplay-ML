import os
import yaml
import wandb
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning.loggers import WandbLogger


def init_wandb(args):
    if 'SLURM_JOB_ID' in os.environ:
        if 'LOCAL_RANK' in os.environ:
            if int(os.environ.get('LOCAL_RANK', 0)) == 0:
                wandb.login(key=args.wandb_key)
                wandb.init(project=args.wandb_project,
                           name=args.wandb_task,
                           config=vars(args),
                           entity=args.wandb_entity,
                           mode=args.wandb_mode)
                wandb_logger = WandbLogger(project=args.wandb_project,
                                        log_model=False,
                                        offline=False)
                logger = [wandb_logger]
            else:
                logger = []
        elif int(os.environ.get('SLURM_PROCID', 0)) == 0:
            wandb.login(key=args.wandb_key)
            wandb.init(project=args.wandb_project,
                       name=args.wandb_task,
                       config=vars(args),
                       entity=args.wandb_entity,
                       mode=args.wandb_mode)
            wandb_logger = WandbLogger(project=args.wandb_project,
                                    log_model=False,
                                    offline=False)
            logger = [wandb_logger]
        else:
            logger = []
    else:
        if int(os.environ.get('LOCAL_RANK', 0)) == 0:
            wandb.login(key=args.wandb_key)
            wandb.init(project=args.wandb_project,
                       name=args.wandb_task,
                       config=vars(args),
                       entity=args.wandb_entity,
                       mode=args.wandb_mode)
            wandb_logger = WandbLogger(project=args.wandb_project,
                                    log_model=False,
                                    offline=False)
            logger = [wandb_logger]
        else:
            logger = []

    return logger


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
