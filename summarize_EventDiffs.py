# -*- coding: utf-8 -*-
"""
Created on Thu May 22 12:36:51 2025

@author: Vicon-OEM
"""
import os
import tkinter as tk
from tkinter import filedialog

main_directory = r'K:\ViconDatabase\Python Code\1. readC3D_local'

def get_c3d_files():
    # Create a Tkinter root window (hidden)
    root = tk.Tk()
    root.withdraw()

    # Ask the user to select a folder
    folder_selected = filedialog.askdirectory(initialdir=main_directory, title="Select a Folder")

    if not folder_selected:
        print("No folder selected.")
        return []

    # Get all .c3d file paths in the folder
    c3d_files = [os.path.join(folder_selected, f) for f in os.listdir(folder_selected) if f.lower().endswith(".c3d")]

    return c3d_files

# Example usage
c3d_files_list = get_c3d_files()
print("Found .c3d files:", c3d_files_list)
