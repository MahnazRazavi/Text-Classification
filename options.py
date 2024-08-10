from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class TextOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Text classification options")


        # PATHS
        self.parser.add_argument("--model_path",
                                 type=str,
                                 help="path to the pre-trained model",
                                 default=os.path.join(file_dir, "model/bert-base-uncased"))
        
        # TOKENISATION options
        self.parser.add_argument("--token_output",
                                type=str,
                                 help="path to the save tokenized data",
                                 default=os.path.join(file_dir, "output/data/token/"))

        
        # TRAINING options
        self.parser.add_argument("--model_output",
                                type=str,
                                 help="path to the trained model",
                                 default=os.path.join(file_dir, "output/model/tcl"))
        
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default=os.path.join(file_dir, "dataset/xenophobia_racism_dataset.csv"))
        
        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=8)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=2e-5)
        self.parser.add_argument("--epsilon",
                                 type=float,
                                 help="epsilon",
                                 default=1e-08)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=20)
        
    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
