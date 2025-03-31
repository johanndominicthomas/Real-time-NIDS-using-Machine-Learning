import pandas as pd
import os
dataset_folder='Datasets'
output_file='combined_cic_ids2017.csv'
def combine_datasets(folder,output_file):
    """
    Combine multiple CSV files into a single CSV file.
    
    Parameters:
        folder (str): Path to the folder containing the dataset files.
        output_file (str): Path to save the combined dataset.
    """
    #List all CSV files are in order
    csv_files= [f for f in os.listdir(folder) if f.endswith('.csv')]
    csv_files.sort()  #Ensure files are processed in order
    
    #Initialize an empty list to store DataFrames
    combined_data=[]
    
    print(f"Found {len(csv_files)} CSV files to combine.")
    
    #Iterate through the files and read them
    for file in csv_files:
	    file_path=os.path.join(folder,file)
	    print(f"Reading {file_path}...")
	    df=pd.read_csv(file_path)
	    combined_data.append(df)
    
    #Concatenate all DataFrames
    combined_df=pd.concat(combined_data,ignore_index=True)
    
    #save combined dataset to a new CSV file
    combined_df.to_csv(output_file,index=False)
    print(f"Combined dataset saved to {output_file}")    
    
#Run the function
if __name__=="__main__":
	combine_datasets(dataset_folder,output_file)
    
