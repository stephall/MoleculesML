#load_ZINC_data.py

# Load modules
import numpy as np
import urllib
import copy
import random
import pandas as pd

def load_from_web(ZINC_tranches_list_file_name, num_tranche_samples, random_seed=42):
    """
    Load sampled tranches from the ZINC data base.

    Args:
        ZINC_tranches_list_file_name (str): Name (path) of the file containing the urls 
            to the tranches of ZINC.
        num_tranche_samples (int): Number of tranches we should sample and load.
        random_seed (int): Random seed for the sampling.
            (Default 42)

    Returns:
        (pandas.DataFrame): DataFrame containing the data.
    """

    # Get the tranche url list from the file
    tranche_url_list = get_tranche_url_list(ZINC_tranches_list_file_name)

    # Get the number of tranches
    num_tranches = len(tranche_url_list)
            
    # Set a random seed
    np.random.seed(random_seed)

    # Sample tranches
    sampled_tranche_number_list = list()
    for sample_index in range(num_tranche_samples):
        print(f"Sample {sample_index+1}/{num_tranche_samples}")
        
        # Sample a tranche and return its content as dictionary
        while True:
            # Sample a tranche number
            sampled_tranche_number = np.random.choice(range(num_tranche_samples))

            # If this tranche was already sampled, continue (sample again)
            if sampled_tranche_number in sampled_tranche_number_list:
                continue

            # Get the url of the sampled tranche
            tranche_url = tranche_url_list[sampled_tranche_number]

            # Try to open the url
            try:
                with urllib.request.urlopen(tranche_url) as url_file:
                    # Generate the content dictionary from the url_file
                    content_dict = generate_content_dict(url_file)
            except urllib.error.HTTPError:
                # If the url is not valid, continue (sample again)
                continue

            # Append the sampled tranche number to the list of sampled tranche numbers
            sampled_tranche_number_list.append(sampled_tranche_number)

            # Break the while loop
            break
        
        # Differ cases depending if it is the first sample or not.
        if sample_index==0:
            # The data dictionary is initialized as a copy of the content dictionary of the first sample
            data_dict = copy.deepcopy(content_dict)
        else:
            # In this case, the entries of the content dictionary are added to the data dictionary
            # Step 1: Check that the keys of the data dictionary and content dictionary are the same
            if set(data_dict.keys())!=set(content_dict.keys()):
                err_msg = f"The keys of the content dictionary of row {row_number} are not the same as for the data dictionary.\n" \
                          f"The keys of the data dictionary are: {data_dict.keys()}" \
                          f"The keys of the content dictionary are: {content_dict.keys()}"
                raise KeyError(err_msg)

            # Step 2: Loop over the keys of both dictionaries and add the lists
            for key in data_dict:
                # Add the list of the content dictionary to the corresponding 
                # list of the data dictionary
                data_dict[key] += content_dict[key]
            
                # Loop over the keys of both dictionaries and add the lists
                for key in data_dict:
                    data_dict[key] += content_dict[key]

    # Map the data dictionary to a pandas data frame
    data_df = pd.DataFrame.from_dict(data_dict)

    return data_df

def generate_content_dict(url_file):
    """
    Generate the content dictionary from the conent of an url.
    
    Args:
        url_file (str): String/Bytestring of the content of the url file.
        
    Returns:
        (dict): Dictionary containing the url's content.
    """
    # Read the content of the url
    url_content = str( url_file.read() )

    # Loop over the rows of the file by splitting the url_content on the newline character
    # Rem: 1) The rows are terminated not only by a newline (\n) but also carriage return (\r) character, 
    #         thus split on both.
    #      2) The first two litterals are "b'" and the last one is "'"
    row_list = url_content[2:].split('\\r\\n')
    for row_number, row_content in enumerate(row_list):
        # Split on tabulators (\t) to obtain the row items
        row_items = row_content.split('\\t')

        # Differ cases where it is the first row, which is the header row or any other row 
        if row_number==0:
            # Use row items of the first row (header) as the keys to define
            # the content dictionary, which will be a dictionary of lists
            keys = row_items
            content_dict = {key: [] for key in keys}
        else:
            # The last row only contains an empty string (acutall the single char "'") due to the splitting, thus skip it.
            if row_number==len(row_list)-1:
                continue

            # Check that the number of row items is equal to the number of keys
            if len(keys)!=len(row_items):
                # if this is the case, display a message and the row.
                print(f"Row number {row_number} has not the correct number of row items and is therefore skipped.")
                print(f"The content of this row is: {row_content}")
                print()
                # Continue to the next loop iteration
                continue

            # Loop over the row items and append them to their corresponding list in the content dictionary
            for key, row_item in zip(keys, row_items):
                # Try to map the row_item to a number.
                try:
                    row_item = float(row_item)
                except ValueError:
                    pass
                content_dict[key].append(row_item)

    return content_dict

def get_tranche_url_list(ZINC_tranches_list_file_name):
    """ 
    Return a list of urls of the different tranches. 
    
    Args:
        tranches_list_file_name (list): Name (path) to the file containing the urls of the ZINC tranches
        
    Returns:
        (list): List of the urls of the tranches.
    """
    # Open the tranches list file and generate a list of tranches with a valid url
    tranche_url_list = list()
    with open(ZINC_tranches_list_file_name, 'r') as tranches_list_file:
        # Each line of the tranches list file corresponds to a url address
        for tranche_url in tranches_list_file.readlines():
            # Append the url
            tranche_url_list.append(tranche_url)
    
    return tranche_url_list
