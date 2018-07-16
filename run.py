from __future__ import absolute_import
from __future__ import print_function

import sys
import os
import argparse
import datetime

import json

from app.nl2prog_meta import NL2Prog_meta
from app.nl2prog import NL2Prog

from pprint import pprint

class Logger(object):

    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    

import tensorflow as tf
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run nl2prog model.')
    
    parser.add_argument('--input-dir', dest='input_dir', action='store', 
                        help='The directory containing trainning input.', 
                        default=os.path.join("..", "..", "nl2prog", "input"))
    
    parser.add_argument('--output-dir', dest='output_dir', action='store', 
                        help='The directory to store the trainning output.', 
                        default=os.path.join(".", "output"))
    
    parser.add_argument('--config', dest='config', action='store', 
                        help='Configuration of the network and parameters.',
                        default=os.path.join(".", "nl2prog.config"))
    
    parser.add_argument('--production', dest='production', action='store_true', 
                        help='Run the network using production parameters.', default=False)
    
    parser.add_argument('--copy-stdout', dest='copy_stdout', action='store_true', 
                        help='Copy stdout to log.', default=True)

    parser.add_argument('--test-model', dest='test_model', action='store',
                        help='test the specified model with provided config.', default=None)

    parser.add_argument('--meta_learning', dest='meta_learning', action='store_true',
                        help='meta_learning.', default=False)

    parser.add_argument('--learning_rate', dest='learning_rate', action='store',
                        help='Learning rate.', default=None)
    parser.add_argument('--meta_learning_rate', dest='meta_learning_rate', action='store',
                        help='meta learning rate.', default=None)
    parser.add_argument('--gradient_clip_norm', dest='gradient_clip_norm', action='store',
                        help='The directory containing trainning input.', default=None)
    parser.add_argument('--num_meta_example', dest='num_meta_example', action='store',
                        help='The directory containing trainning input.', default=None)
    parser.add_argument('--num_layers', dest='num_layers', action='store',
                        help='num_layers.', default=None)
    parser.add_argument('--value_based_loss', dest='value_based_loss', action='store',
                        help='value_based_loss: sum_vloss, max_vloss, ploss', default=None)


    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    production = args.production
    model_to_test = args.test_model
    meta_learning = args.meta_learning
    
    with open(args.config, "r") as f:
        config = json.load(f)
        if production:
            config["hyper_param"] = config["production_hyper_param"]
        else:
            config["hyper_param"] = config["dev_hyper_param"]
            # fix seeds for testing
            np.random.seed(1)
            tf.set_random_seed(1)
        
        config.pop("dev_hyper_param")
        config.pop("production_hyper_param")

    if args.learning_rate:
      config["hyper_param"]['learning_rate'] = float(args.learning_rate)
    if args.meta_learning_rate:
      config["hyper_param"]['meta_learning_rate'] = float(args.meta_learning_rate)
    if args.gradient_clip_norm:
      config["hyper_param"]['gradient_clip_norm'] = float(args.gradient_clip_norm)
    if args.num_meta_example:
      config["hyper_param"]['num_meta_example'] = int(args.num_meta_example)
    if args.num_layers:
      config["hyper_param"]['num_layers'] = int(args.num_layers)
    if args.value_based_loss:
      config['value_based_loss'] = args.value_based_loss

    if args.copy_stdout:
        sys.stdout = Logger(os.path.join(output_dir, "stdout.log"))

    print("[OK] Using tensorflow version {}".format(tf.__version__))

    if meta_learning:
      NL2Prog_meta.run_wikisql(input_dir, output_dir, config, model_to_test)
    else:
      NL2Prog.run_wikisql(input_dir, output_dir, config, model_to_test)
