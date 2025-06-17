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
csv_files = glob.glob(f"K:\\ViconDatabase\\Python Code\\Gait_Event_check\\Data\\*{search_string}*.csv")
dataframes = [pd.read_csv(file).assign(source=file.split('\\')[-1][0:6]) for file in csv_files]

# Merge the dataframes
merged_df = pd.concat(dataframes, ignore_index=False)  # Merge all into one dataframe
# drop the 'unnamed' column
merged_df.drop(columns=['Unnamed: 0'], inplace=True)

# --------------------------- Plot SHR vs. All TD data ------------------------
# -----------------------------------------------------------------------------
cols = merged_df.columns
# datadf = merged_df.drop(columns=[cols[0], 'Subject']).copy()

# melt dataframe to long-format
comp = 'Gold_Marker'
melt_df = merged_df.melt(id_vars=['source'], 
                      value_vars=comp, # column with marker data
                      var_name='Comparison', # column with marker names
                      value_name='Event Diff')

# Plot summary TD vs SHR data
sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

SH_TD_plot = sns.barplot(
    data=melt_df, 
    x="source", y='Event Diff', 
    ax=axes[0],
    palette='viridis',
    order=['SH_off','TD_off','SH_str','TD_str'],
    hue="source",
)

# --------------------------- Plot Invididual Site Data -----------------------
# -----------------------------------------------------------------------------
# Make 'Site' column using the first two elements of the 'Subject' column name
merged_df['Site'] = merged_df['Subject'].str.slice(0,2)
# drop 'Subject' column
merged_df.drop(columns=['Subject'], inplace=True)

# Update site names
merged_df.loc[merged_df['Site'] == 'Pa', 'Site'] = 'SH'
merged_df.loc[merged_df['Site'] == 'TD', 'Site'] = 'SC'

# Make only foot off events dataframe, then melt
off_site_df = merged_df[(merged_df['source'] == 'SH_off') | (merged_df['source'] == 'TD_off')].copy()
melt_Offdf = off_site_df.melt(id_vars='Site',
                              value_vars=['Gold_Marker','Force_Marker','Gold_Force'],
                              var_name='Comparison',
                              value_name='Event Diff')

siteOff_plot = sns.barplot(
    data=melt_Offdf[melt_Offdf['Comparison'] == comp], 
    x="Site", y='Event Diff', 
    ax=axes[1],
    palette='viridis',
    hue="Site",
)

# Make only foot strike events dataframe, then melt
str_site_df = merged_df[(merged_df['source'] == 'SH_str') | (merged_df['source'] == 'TD_str')].copy()
melt_Strdf = str_site_df.melt(id_vars='Site',
                              value_vars=['Gold_Marker','Force_Marker','Gold_Force'],
                              var_name='Comparison',
                              value_name='Event Diff')

siteStr_plot = sns.barplot(
    data=melt_Strdf[melt_Strdf['Comparison'] == comp], 
    x="Site", y='Event Diff', 
    ax=axes[2],
    palette='viridis',
    hue="Site",
)

# Set subplot titles
axes[0].set_title('Summary Event Diff')
axes[1].set_title('Site Foot Off Event Diff')
axes[2].set_title('Site Foot Strike Event Diff')

# Set y-axis limits
axes[0].set_ylim(0,2.5)
axes[1].set_ylim(0,2.5)
axes[2].set_ylim(0,2.5)

fig.suptitle(f"Event Difference Comparison {comp}", fontsize=16)
plt.tight_layout()
plt.show()

png_filename = f'Event_Comparison_{comp}.png'
save_png_path = os.path.join('K:\\ViconDatabase\\Python Code\\Gait_Event_check\\', png_filename)
fig.savefig(save_png_path, dpi=300, bbox_inches='tight')





