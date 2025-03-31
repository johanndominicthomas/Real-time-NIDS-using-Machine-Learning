import pandas as pd 

#Path to combined dataset
combined_file='combined_cic_ids2017.csv'
#Path to save the 10% subset
subset_file='combined_cic_ids2017_30percent.csv'

#Read the full dataset
df=pd.read_csv(combined_file)

#Sample 10% of the dataset
df_sampled=df.sample(frac=0.3,random_state=42) 
#Random state ensures same 10% is produced when code is rerun

#Save the sampled data to new file
df_sampled.to_csv(subset_file,index=False)

print(f"30% of the dataset saved to {subset_file}")
