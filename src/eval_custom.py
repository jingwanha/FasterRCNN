#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train.py

import argparse
import logging
import six
import os
assert six.PY3, "This example requires Python 3!"

from tensorpack import *
from tensorpack.tfutils import get_model_loader, collect_env_info
from tensorpack.tfutils.common import get_tf_version_tuple
from tensorpack.utils import logger

from tensorpack.predict import MultiTowerOfflinePredictor, OfflinePredictor, PredictConfig


from dataset import DatasetRegistry, register_custom_data
from config import config as cfg
from config import finalize_configs
from data import get_eval_dataflow,get_train_dataflow
from eval import EvalCallback, multithread_predict_dataflow
from modeling.generalized_rcnn import ResNetC4Model, ResNetFPNModel

try:
    import horovod.tensorflow as hvd
except ImportError:
    pass

logger.setLevel(logging.DEBUG)  # set level of logger

def do_evaluate(pred_config, output_file):
    num_tower = max(cfg.TRAIN.NUM_GPUS, 1)
    graph_funcs = MultiTowerOfflinePredictor(pred_config, list(range(num_tower))).get_predictors()

    for dataset in cfg.DATA.VAL.to_dict().keys():
        logger.info("Evaluating {} ...".format(dataset))
        dataflows = [
            get_eval_dataflow(dataset, shard=k, num_shards=num_tower)
            for k in range(num_tower)]
        all_results = multithread_predict_dataflow(dataflows, graph_funcs)
        
        # output = os.path.splitext(output_file)[0] + '-' + dataset + '.json'
        # DatasetRegistry.get(dataset).eval_inference_results(all_results, output)
        mAP_result=DatasetRegistry.get(dataset).eval_inference_results(all_results)
        
        import json
        with open(output_file, 'w') as f:
            json.dump(mAP_result, f)

        

if __name__ == '__main__':
    # "spawn/forkserver" is safer than the default "fork" method and
    # produce more deterministic behavior & memory saving
    # However its limitation is you cannot pass
    # a lambda function to subprocesses.
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='load a model to start training from. '
                                       'Can overwrite BACKBONE.WEIGHTS')
    parser.add_argument('--config', help="config json file path")
    parser.add_argument('--logdir',
                        help='log directory',
                        default='train_log/maskrcnn')
    parser.add_argument('--output_dir', help="directory to save output images")


    args = parser.parse_args()
    
    if args.config:
        cfg.update_from_json(args.config)
        cfg.DATA.CLASS_NAMES = ['BG'] + cfg.DATA.CLASS_NAMES
    # register datasets
    register_custom_data()
    
    finalize_configs(is_training=True)
    logger.debug(cfg)

    # Create model
    MODEL = ResNetFPNModel() if cfg.MODE_FPN else ResNetC4Model()
    
    predcfg = PredictConfig(
        model=MODEL,
        session_init=get_model_loader(args.load),
        input_names=MODEL.get_inference_tensor_names()[0],
        output_names=MODEL.get_inference_tensor_names()[1])
    
    do_evaluate(predcfg, args.output_dir)