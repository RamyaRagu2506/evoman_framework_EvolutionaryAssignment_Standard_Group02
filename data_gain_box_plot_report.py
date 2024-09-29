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
        es_data['Method'] = 'Algorithm Without FS'  
        fs_data['Method'] = 'Algorithm with FS'  

        # Combine the two datasets for plotting
        combined_data = pd.concat([es_data[['Iteration', 'Avg_Individual_Gain', 'Method']], fs_data[['Iteration', 'Avg_Individual_Gain', 'Method']]])

        # Create the plot
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Method', y='Avg_Individual_Gain', data=combined_data,hue='Method',palette=['blue','red'])
        
        # Add titles and labels
        plt.title('Gain Plot for Enemy 3')
        plt.xlabel('Algorithms')
        plt.ylabel('Gain')
        plt.savefig('test/gain_plot_enemy3')

        # Show the plot
        plt.show()
    except Exception as e:
        print(f"An error occurred: {e} check parameters involved in gain plot")

# es and fs file paths 
es_file_path = 'es_enemy3/simulation_results_es.xlsx'  
fs_file_path = 'fs_enemy3/simulation_results_fs.xlsx'  

# Create the gain plot
gain_plot(es_file_path, fs_file_path)
