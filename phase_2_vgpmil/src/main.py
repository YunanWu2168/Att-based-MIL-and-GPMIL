from __future__ import print_function
import yaml
import os
import argparse
import numpy as np
import pandas as pd
import timeit
from vgpmil.helperfunctions import RBF
from vgpmil.vgpmil import vgpmil
from typing import Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from loading import load_dataframe, load_cnn_predictions, get_bag_level_information
from metrics import Metrics
from tsne_visualization import visualize_and_save

def initialize_model(config):
    vgpmil_config = config['vgpmil']
    kernel = RBF(lengthscale=vgpmil_config['kernel_length_scale'], variance=vgpmil_config['kernel_variance'])
    vgpmil_model = vgpmil(kernel=kernel,
                           num_inducing=int(vgpmil_config['inducing_points']),
                           max_iter=int(vgpmil_config['iterations']),
                           normalize=bool(vgpmil_config['normalize']),
                           verbose=bool(vgpmil_config['verbose']))


    return vgpmil_model

def train(config: Dict, vgpmil_model: vgpmil = None):
    print('Training..')
    train_df = pd.read_csv(config['path_train_df'])
    print('Loaded training dataframe. Number of instances: ' + str(len(train_df)))
    features, bag_labels_per_instance, bag_names_per_instance, instance_labels = load_dataframe(train_df, config)
    bag_features, bag_labels, bag_names = get_bag_level_information(features, bag_labels_per_instance, bag_names_per_instance)

    if config['tsne']['visualize']:
        visualize_and_save(features=features, instance_labels=instance_labels, config=config)
    print('Train VGPMIL')
    vgpmil_model.train(features, bag_labels_per_instance, bag_names_per_instance, Z=None, pi=None, mask=None)

def test(config: Dict, vgpmil_model: vgpmil = None):
    print('Testing..')
    test_df = pd.read_csv(config['path_test_df'])
    print('Loaded test dataframe. Number of instances: ' + str(len(test_df)))
    features, bag_labels_per_instance, bag_names_per_instance, instance_labels = load_dataframe(test_df, config)
    bag_features, bag_labels, bag_names = get_bag_level_information(features, bag_labels_per_instance, bag_names_per_instance)

    metrics_calculator = Metrics(instance_labels, bag_labels, bag_names, bag_names_per_instance)

    print('Test VGPMIL')
    start = timeit.timeit()
    instance_predictions, bag_predictions = vgpmil_model.predict(features, bag_names_per_instance, bag_names)
    end = timeit.timeit()
    print('Average runtime per bag: ', str((end - start) / bag_predictions.size))
    metrics_calculator.calc_metrics(instance_predictions, bag_predictions, 'vgpmil')
    cnn_predictions, bag_cnn_predictions, bag_cnn_probability = load_cnn_predictions(test_df, config)
    metrics_calculator.calc_metrics(cnn_predictions, bag_cnn_probability, 'cnn')

    metrics_calculator.write_to_file(config)

def main():
    parser = argparse.ArgumentParser(description="Cancer Classification")
    parser.add_argument("--config", "-c", type=str, default="./config.yaml",
                        help="Config path (yaml file expected) to default config.")
    args = parser.parse_args()
    with open(args.config) as file:
        config = yaml.full_load(file)
    vgpmil_model = initialize_model(config)
    train(config, vgpmil_model)
    test(config, vgpmil_model)




if __name__ == "__main__":
    main()
