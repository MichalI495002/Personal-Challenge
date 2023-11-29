import pandas as pd
import matplotlib.pyplot as plt

def plot_graph(file_path, width=600, height=400):
    try:
        # Read data from file path into a DataFrame
        dataframe = pd.read_csv(file_path)

        # Debugging: Print the first few rows of the DataFrame
        # print(dataframe.head())

        # Retrieve the names of the first two columns
        columns = dataframe.columns
        if len(columns) < 2:
            raise ValueError("The file must contain at least two columns.")

        x_column = columns[0]
        y_column = columns[1]

        # Create a plot
        dpi = 100  # Set the dots per inch to render
        fig_width = width / dpi  # Convert width from pixels to inches
        fig_height = height / dpi  # Convert height from pixels to inches

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        fig.patch.set_facecolor("#2E2E2E")
        ax.set_facecolor("#2E2E2E")
        
        # Customize the graph appearance
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.yaxis.label.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.title.set_color('white')
        ax.grid(color='gray', linestyle='--', linewidth=0.5)
        ax.set_title('Modern Dark Themed Graph')

        # Plot the data
        ax.plot(dataframe[x_column], dataframe[y_column])

        return fig
    except Exception as e:
        raise Exception(f"An error occurred: {e}")

