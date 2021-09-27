# This script calculates the dataset statistics of the train and test dataframe specified in the ./config.yaml

import yaml
import pandas as pd
import numpy as np


def main():
    with open('config.yaml') as file:
        config = yaml.full_load(file)
    train_df = pd.read_csv(config['path_train_df'])
    test_df = pd.read_csv(config['path_test_df'])
    dict = {'Train': train_df, 'Test': test_df}
    for name, df in dict.items():
        bag_sizes = []

        print(name)
        print('Instance stats:')
        print('total: ', len(df))
        if config['col_instance_label'] in df:
            pos_inst = np.sum(df[config['col_instance_label']] == 1)
            print('positive: ', pos_inst)
            print('negative: ', len(df) - pos_inst)

        print('Bag stats:')
        bags = np.unique(df[config['col_bag_name']])
        print('total: ', bags.size)
        pos_bags = 0

        for bag in bags:
            df_bag = df.loc[df[config['col_bag_name']] == bag]
            label = np.unique(df_bag[config['col_bag_label']])
            bag_sizes.append(len(df_bag))
            if label == 1:
                pos_bags = pos_bags + 1
        bag_sizes = np.array(bag_sizes)

        print('average_bag_size: ', np.mean(bag_sizes))
        print('min_bag_size: ', np.min(bag_sizes))
        print('max_bag_size: ', np.max(bag_sizes))
        print('positive: ', pos_bags)
        print('negative: ', bags.size - pos_bags)
        print('\n')







if __name__ == "__main__":
    main()