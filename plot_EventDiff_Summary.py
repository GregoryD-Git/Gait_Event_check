# -*- coding: utf-8 -*-
"""
Created on Fri May 23 12:43:44 2025

@author: Vicon-OEM
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os

# Load CSV files into dataframes
search_string = 'data'
csv_files = glob.glob(f"K:\\ViconDatabase\\Python Code\\Gait_Event_check\\*{search_string}*.csv")
dataframes = [pd.read_csv(file).assign(source=file.split('\\')[-1][0:6]) for file in csv_files]
# [print(file.split('\\')) for file in csv_files]

# Merge the dataframes
merged_df = pd.concat(dataframes, ignore_index=True)  # Merge all into one dataframe
cols = merged_df.columns
datadf = merged_df.drop(columns=[cols[0], 'Subject']).copy()

# melt dataframe to long-format
df_cols = [col for col in cols if 'Gold_Marker' in col]
melt_df = datadf.melt(id_vars=['source'], 
                      value_vars=df_cols, # column with marker data
                      var_name='Comparison', # column with marker names
                      value_name='Event Diff')

# melt_df["event"] = melt_df["source"].apply(lambda x: "foot_off" if "_off" in x else "foot_strike")

# Plot summary data
sns.set_style("whitegrid")

my_plot = sns.catplot(
    data=melt_df, 
    x="source", y='Event Diff', 
    order=['SH_off','TD_off','SH_str','TD_str'],
    hue="source",
    kind="bar"
)

# Show the plot
plt.title("Summary of Event differences")
plt.show()

png_filename = 'Event_SHR-TD_Comparison.png'
save_png_path = os.path.join('K:\\ViconDatabase\\Python Code\\Gait_Event_check\\', png_filename)
my_plot.savefig(save_png_path, dpi=300, bbox_inches='tight')