# Author: Thomas Herold
# File: unifyCSV.py

import os # Provides functions for file system interaction
import shutil # Provides functions for copying, moving, and removing directories/files
import glob # Provides functions for finding files
import re # Provides regex utilities for robust name normalization

# Dynamically map folder names to categories (case insensitive)
def get_category_from_folder(folder_name):
    # Normalize aggressively so minor spelling/spacing differences still match.
    # Examples handled: "Take Off", "take_off", "B ackward", "foreward", "fowward", etc.
    raw = (folder_name or "").lower()
    normalized = re.sub(r"[^a-z]+", "", raw)  # keep letters only; drop spaces/underscores/punct/digits

    if any(key in normalized for key in ("takeoff", "takoff", "takeof")):
        return "takeoff"
    elif any(key in normalized for key in ("backward", "backwards", "backard", "bakward")):
        return "backward"
    elif "right" in normalized:
        return "right"
    elif "left" in normalized:
        return "left"
    elif any(key in normalized for key in ("forward", "foreward", "fowward")):
        return "forward"
    elif "landing" in normalized:
        return "landing"
    else:
        print(f"Category not found: {folder_name}") # Print message to indicate no category was found
        return None  # Return none to indicate no matching category was found

# Process the directory with the BCI data
def move_files_to_categories(base_dir):
    for group in os.listdir(base_dir): # Loop over each group (not static) subdirectory
        group_path = os.path.join(base_dir, group)
        if not os.path.isdir(group_path): # Skip if it is not a directory
            continue

        for subdir in os.listdir(group_path): # Loop over each individual/test (not static) subdirectory
            subdir_path = os.path.join(group_path, subdir)
            if os.path.isdir(subdir_path):

                for folder in os.listdir(subdir_path): # Loop over each subdirectory (landing, forward, etc.)
                    folder_path = os.path.join(subdir_path, folder)
                    if os.path.isdir(folder_path): 
                        category = get_category_from_folder(folder) # Determine the folder category for unifying (landing, forward, left, etc.)
                        if category: # Only continue if a valid category is identified
                            print(f"Processing category: {category}")
                            category_path = os.path.join(base_dir, category)
                            if not os.path.exists(category_path): # Create the category directory if it does not exist
                                os.makedirs(category_path)
                            
                            csv_files = glob.glob(os.path.join(folder_path, "*.csv")) # Get all .csv files in the folder
                            
                            for csv_file in csv_files:
                                filename = os.path.basename(csv_file) # Extract the filename from the full path
                                destination = os.path.join(category_path, filename) # Define the destination path for the CSV file

                                if not os.path.exists(destination):  # Avoid duplicates in target subdirectory
                                    print(f"Moving: {csv_file} to {destination}")
                                    shutil.move(csv_file, destination) # Move .csv file to the target category folder
                                else:
                                    print(f"Removing duplicate file: {filename}")
                                    os.remove(csv_file) # Delete the duplicate .csv file

                            txt_files = glob.glob(os.path.join(folder_path, "*.txt")) # Find .txt files
                            for txt_file in txt_files:
                                print(f"Deleting .txt file: {txt_file}")
                                os.remove(txt_file)  # Delete the .txt file

                            if not os.listdir(folder_path): # Check if subdirectory is empty
                                print(f"Removing empty subdirectory: {folder_path}")
                                os.rmdir(folder_path) # Remove subdirectory

                if not os.listdir(subdir_path): # Check if subdirectory is empty
                    print(f"Removing empty subdirectory: {subdir_path}")
                    os.rmdir(subdir_path) # Remove subdirectory

        if not os.listdir(group_path): # Check if subdirectory is empty
            print(f"Removing empty group directory: {group_path}")
            os.rmdir(group_path) # Remove subdirectory

    print(f"{base_dir} directory processed, CSV files unified!") # Program completion message

if __name__ == "__main__":
    base_directory = "data"  # Base directory to start from: \data\. This should be in the same directory as the Python script.
    move_files_to_categories(base_directory) # Function call to do all of the data processing
