# -*- coding: utf-8 -*-
"""
Created on Thu May 22 12:36:51 2025

@author: Vicon-OEM
"""
import os
import tkinter as tk
from tkinter import filedialog
from check_EventTiming import get_EventDiffs as evd
import pandas as pd

main_directory = r'K:\ViconDatabase\Python Code\Gait_Event_check'
x_thresh = 0.35
data_source = 'SH'

def get_c3d_files():
    # Create a Tkinter root window (hidden)
    root = tk.Tk()
    # root.withdraw()

    # Ask the user to select a folder
    folder_selected = filedialog.askdirectory(initialdir=main_directory, title="Select a Folder")

    if not folder_selected:
        print("No folder selected.")
        return []

    # Get all .c3d file paths in the folder
    c3d_files = [os.path.join(folder_selected, f) for f in os.listdir(folder_selected) if f.lower().endswith(".c3d")]
    
    root.destroy()
    
    return c3d_files

# Example usage
c3d_files_list = get_c3d_files()
dir_mk = 'X' # a/p marker

save_folderpath = r'K:\ViconDatabase\Python Code\2. Event timing plots'
off_df = pd.DataFrame()
str_df = pd.DataFrame()

for filenamepath in c3d_files_list: 
    df1, df2 = evd(filenamepath, dir_mk, save_folderpath, x_thresh, data_source)
    off_df = pd.concat([off_df, df1], axis=0)
    str_df = pd.concat([str_df, df2], axis=0)

off_summary = off_df.describe()
str_summary = str_df.describe()

# Define the output file path
off_data_file = os.path.join(main_directory, f'{data_source}_off_data_threshold-{x_thresh}.csv')  # Update with your desired path
str_data_file = os.path.join(main_directory, f'{data_source}_str_data_threshold-{x_thresh}.csv')
off_summary_file = os.path.join(main_directory, f'{data_source}_off_summary_threshold-{x_thresh}.csv')  
str_summary_file = os.path.join(main_directory, f'{data_source}_str_summary_threshold-{x_thresh}.csv')

# Save to CSV
off_df.to_csv(off_data_file)
str_df.to_csv(str_data_file)
off_summary.to_csv(off_summary_file)
str_summary.to_csv(str_summary_file)



