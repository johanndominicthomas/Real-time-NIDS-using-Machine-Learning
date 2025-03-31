import pandas as pd

# Load your preprocessed CSV.
df = pd.read_csv("test_preprocessed.csv")

# Filter out rows with label 0 (assuming 0 corresponds to BENIGN).
attack_df = df[df['Label'] != 0].copy()

print("Number of attack flows detected:", attack_df.shape[0])
print("Unique attack labels in attack flows:", attack_df['Label'].unique())

# Now drop the Label column since the model doesn't require it.
attack_df_no_labels = attack_df.drop(columns=['Label'])

# Save the DataFrame without the label column to a new CSV.
attack_df_no_labels.to_csv("attack_flows_no_labels.csv", index=False)
print("Created 'attack_flows_no_labels.csv' containing only attack features (without labels).")
