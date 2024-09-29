import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_excel_data(file_path):
    return pd.read_excel(file_path)

def gain_plot(es_file, fs_file):
    try: 
        # Load the Excel files
        es_data = load_excel_data(es_file)
        fs_data = load_excel_data(fs_file)

        # Add a label to each dataset to differentiate between ES and FS
        es_data['Method'] = 'Memetic ES'  
        fs_data['Method'] = 'Memetic with FS'  

        # Combine the two datasets for plotting
        combined_data = pd.concat([es_data[['Run', 'Individual_Gain', 'Method']], fs_data[['Run', 'Individual_Gain', 'Method']]])

        # Create the plot
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Method', y='Individual_Gain', data=combined_data)
        
        # Add titles and labels
        plt.title('Gain Plot for Enemy 4')
        plt.xlabel('Algorithms')
        plt.ylabel('Gain')
        plt.savefig('test/gain_plot_enemy4')

        # Show the plot
        plt.show()
    except Exception as e:
        print(f"An error occurred: {e} check parameters involved in gain plot")

# es and fs file paths 
es_file_path = 'test/es_enemy_4.xlsx'  
fs_file_path = 'test/fs_enemy_4.xlsx'  

# Create the gain plot
gain_plot(es_file_path, fs_file_path)
