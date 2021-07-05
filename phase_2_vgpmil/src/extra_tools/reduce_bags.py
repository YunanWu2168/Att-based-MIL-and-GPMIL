import yaml
import os
import pandas as pd
import numpy as np

path_test_df = './CQ500/input/final/test/CQ500_final_4.csv'
# path_test_df = './CQ500/input/without_attention_factor/CQ500_8.csv'
out_path = './CQ500/input/final/test_sec_dc/'
max_bag_size = 50
reduced_bag_size = 50

second_datacenter_df = './CQ500/input/without_attention_factor/CQ500_8.csv'
second_datacenter_only = True


def main():
    with open('config.yaml') as file:
        config = yaml.full_load(file)
    df = pd.read_csv(path_test_df)
    sd_df = pd.read_csv(second_datacenter_df)
    if config['col_instance_label'] in df:
        print('Instance stats:')
        print('total: ', len(df))
        pos_inst = np.sum(df[config['col_instance_label']] == 1)
        print('positive: ', pos_inst)
        print('negative: ', len(df) - pos_inst)

    print('Bag stats:')
    bags = np.unique(df[config['col_bag_name']])
    sd_bags = np.unique(sd_df[config['col_bag_name']])
    print('total: ', bags.size)
    pos_bags = 0
    new_df = pd.DataFrame()

    for bag in bags:
        df_bag = df.loc[df[config['col_bag_name']] == bag]
        label = np.unique(df_bag[config['col_bag_label']])
        bag_size = len(df_bag)
        if bag_size > max_bag_size:
            r = np.arange(0,reduced_bag_size) * bag_size/reduced_bag_size
            ids = np.round(r).astype(int)
            new_df_bag = df_bag.iloc[ids]
        else:
            new_df_bag = df_bag
        if second_datacenter_only and not bag in sd_bags:
            new_df_bag = pd.DataFrame()

        new_df = pd.concat([new_df, new_df_bag])

        if label == 1:
            pos_bags = pos_bags + 1
    os.makedirs(out_path, exist_ok=True)
    new_df.to_csv(out_path + os.path.basename(path_test_df))
    print('positive: ', pos_bags)
    print('negative: ', bags.size - pos_bags)
    print('\n')

if __name__ == "__main__":
    main()