# config.py
# Global variables

# Written by Diane J. Cook, Washington State University.

# Copyright (c) 2021. Washington State University (WSU). All rights reserved.
# Code and data may not be used or distributed without permission from WSU.


import argparse
import copy


class Config:

    def __init__(self):
        """ Constructor
        """
        self.num_residents = 2
        self.data_filename = "data"
        self.data = []
        self.sequence = []
        self.sensornames = []
        self.plot = True
        self.time_threshold = 1800  # elapsed time threshold in seconds
        self.distance_threshold = 0.5
        self.closeness_threshold = 0.0001
        self.save_model = False
        self.plot_embeddings = False
        self.load_model = False

    def set_parameters(self):
        """ Set parameters according to command-line args list.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('--num_residents',
                            dest='num_residents',
                            type=int,
                            default=self.num_residents,
                            help=('Specify the number of residents in the environment, default={}'
                                  .format(self.num_residents)))
        parser.add_argument('--save_model',
                            dest='save_model',
                            default=self.save_model,
                            action='store_true',
                            help=('Save the embeddings model to a file for a later use,  '
                                  'default={}'.format(self.save_model)))
        parser.add_argument('--load_model',
                            dest='load_model',
                            default=self.load_model,
                            action='store_true',
                            help=('Load the embeddings model from a saved file,  '
                                  'default={}'.format(self.load_model)))
        parser.add_argument('--plot_embeddings',
                            dest='plot_embeddings',
                            default=self.plot_embeddings,
                            action='store_true',
                            help=('Visualize the generated embeddings with a 3D graph and '
                                  '3D animation, default={}'.format(self.plot_embeddings)))
        parser.add_argument('--data',
                            dest='data',
                            type=str,
                            default=self.data_filename,
                            help=('Data file for MR to process, default={}.'
                                  .format(self.data_filename)))
        args = parser.parse_args()

        self.num_residents = args.num_residents
        self.save_model = args.save_model
        self.load_model = args.load_model
        self.plot_embeddings = args.plot_embeddings
        self.data_filename = args.data
        files = list([self.data_filename])

        return files
