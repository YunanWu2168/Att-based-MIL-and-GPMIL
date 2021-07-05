# This script calculates the mean metrics of the final vgpmil model for all .csv metric files in one folder

import glob
import pandas as pd


#
input_dir = './RSNA/output/without_attention_layer_final/'
out_file = './RSNA/output/without_attention_layer_final/mean.csv'

# input_dir = './RSNA/output/final/'
# out_file = './RSNA/output/final/mean.csv'

files = glob.glob(input_dir + "*.csv")

gp_df = pd.DataFrame()
cnn_df = pd.DataFrame()

for file in glob.glob(input_dir + "*.csv"):
    new_df = pd.read_csv(file, index_col=0)
    new_gp_df = new_df['vgpmil'].to_frame()
    new_cnn_df = new_df['cnn'].to_frame()
    gp_df = pd.concat([new_gp_df, gp_df], axis=1)
    cnn_df = pd.concat([new_cnn_df, cnn_df], axis=1)

gp_df_mean = gp_df.mean(axis=1).round(decimals=3)
gp_df_std = gp_df.std(axis=1).round(decimals=3)
cnn_df_mean = cnn_df.mean(axis=1).round(decimals=3)
cnn_df_std = cnn_df.std(axis=1).round(decimals=3)

out_df = pd.concat([gp_df_mean, gp_df_std, cnn_df_mean, cnn_df_std], axis=1).rename(columns={0: 'vgpmil_mean', 1: 'vgpmil_std', 2: 'cnn_mean',3: 'cnn_std'})
out_df.to_csv(out_file)

