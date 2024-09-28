import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_excel_data(file_path):
    """
    Loads the data from an Excel file into a Pandas DataFrame.
    
    Args:
    - file_path (str): Path to the Excel file.
    
    Returns:
    - pd.DataFrame: DataFrame containing the data from the Excel file.
    """
    return pd.read_excel(file_path)

def create_gain_plot(es_file, fs_file):
    """
    Creates a gain plot comparing the individual gains from ES (Evolution Strategy) and FS (Fitness Sharing) runs.
    
    Args:
    - es_file (str): Path to the Excel file for ES results.
    - fs_file (str): Path to the Excel file for FS results.
    
    Returns:
    - None
    """
    # Load the data from both files
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

# Example usage:
es_file_path = 'test/es_enemy_4.xlsx'  # Replace with actual path to the ES file
fs_file_path = 'test/fs_enemy_4.xlsx'  # Replace with actual path to the FS file

# Create the gain plot
create_gain_plot(es_file_path, fs_file_path)
