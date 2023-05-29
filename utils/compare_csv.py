import csv

import pandas as pd

def compare_csv(csv1_file, column1, csv2_file, column2, limit=10000):
    # Read the usernames from csv1 into a pandas dataframe
    csv1_df = pd.read_csv(csv1_file)
    csv1_df = csv1_df.head(limit)
    # Read the usernames from csv2 into another pandas dataframe
    csv2_df = pd.read_csv(csv2_file)
    csv2_df = csv2_df.head(limit)

    # Find the usernames that are in csv2 but not in csv1
    new_usernames = set(csv2_df[column2]) - set(csv1_df[column1])

    visible_users_to_run = pd.DataFrame(new_usernames, columns=['users'])
    
    visible_users_already_done = pd.DataFrame(set(csv2_df[column2]) - set(visible_users_to_run['users']))

    return visible_users_to_run, visible_users_already_done


path = 'paris/'
file1 = path+'botometer_results.csv'
file2 = path+'Visible_Users.csv'

# This will print a set of usernames that are in file2 but not in file1
visible_users_to_run, visible_users_already_done = compare_csv(file1, 'user.user_data.screen_name', file2, 'Target', limit=10000)

visible_users_to_run.to_csv(path+'visible_users_to_run.csv', index=False)
visible_users_already_done.to_csv(path+'visible_users_present_in_botometer_results.csv', index=False)