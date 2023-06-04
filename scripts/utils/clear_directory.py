import os

def clear_directory(directory_path):
    # Iterate over all files and subdirectories in the directory
    for file_name in os.listdir(directory_path):
        # Construct the full path to the file or subdirectory
        file_path = os.path.join(directory_path, file_name)
        # Check if the file is a directory
        if os.path.isdir(file_path):
            # If the file is a directory, recursively clear it
            clear_directory(file_path)
            # Once the directory is empty, remove it
            os.rmdir(file_path)
        else:
            # If the file is a regular file, remove it
            os.remove(file_path)