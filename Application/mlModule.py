import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import seaborn as sns



def generate_graphs(csv_file_path,width=600, height=400, x_labels_freq=15):
    # Load data
    data = pd.read_csv(csv_file_path)
    
    # Convert 'Date' to datetime for better plotting
    data['Date'] = pd.to_datetime(data['Date'])


    # Define a function for creating and saving each graph
    def create_graph(column, title):
        # DPI and size settings
        dpi = 70  # Set the dots per inch to render
        fig_width = width / dpi  # Convert width from pixels to inches
        fig_height = height / dpi  # Convert height from pixels to inches

        fig, ax = plt.subplots()
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
        ax.set_title(title)

        # Setting the frequency of the x-axis labels
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=x_labels_freq))  # Use integer for interval
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        # Plotting the data
        if column == 'Sales':
            sns.lineplot(x='Date', y=column, data=data, ax=ax)
        else:
            sns.lineplot(x='Date', y=column, data=data, ax=ax)

        return fig

    # Creating and saving each graph
    sales_graph_path = create_graph('Sales', 'Sales Over Time')
    inflation_graph_path = create_graph('Inflation', 'Inflation Over Time')
    unemployment_graph_path = create_graph('Unemployment', 'Unemployment Over Time')
    temperature_graph_path = create_graph('Temperature', 'Temperature Over Time')

    return sales_graph_path, inflation_graph_path, unemployment_graph_path, temperature_graph_path

def format_plot(ax, title, x_labels_freq):
    # Set the background color of the plot
    ax.set_facecolor("#2E2E2E")
    fig = plt.gcf()  # Get the current figure
    fig.set_facecolor("#2E2E2E")  # Set the facecolor of the figure

    # Remove the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set the color of the bottom and left spines to white
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')

    # Set the tick parameters to ensure ticks are white
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    # Set the labels to be white
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')

    # Set the title of the axes and its color
    ax.title.set_color('white')
    ax.set_title(title)

    # Set the grid color and style
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    # Set the aspect of the plot to auto
    ax.set_aspect('auto')

    # Remove padding and margins
    fig.tight_layout(pad=0)  # Remove any padding from the layout
    ax.margins(0)  # Remove any margins

    # Set the x-axis and y-axis tick colors
    for tick in ax.get_xticklabels():
        tick.set_color('white')
    for tick in ax.get_yticklabels():
        tick.set_color('white')

    # If x_labels_freq is specified, adjust the x-axis label frequency
    if x_labels_freq:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=x_labels_freq))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    return ax

   


def use_linear_regressor(csv_file_path, width=800, height=600, x_labels_freq=10):
    # Load and preprocess the data
    dataframe = pd.read_csv(csv_file_path)
    dataframe['Date'] = pd.to_datetime(dataframe['Date'])

    # Feature Engineering: Extract year and month from 'Date'
    dataframe['Year'] = dataframe['Date'].dt.year
    dataframe['Month'] = dataframe['Date'].dt.month

    # Define the features (X) and the target (y)
    X = dataframe[['Year', 'Month', 'Inflation', 'Unemployment', 'Temperature']]  # Example features
    y = dataframe['Sales']

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Linear Regression model
    linear_regressor = LinearRegression()
    linear_regressor.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = linear_regressor.predict(X_test)

    # Predict future sales
    future_dates = pd.date_range(start=dataframe['Date'].max() + pd.DateOffset(1), periods=18, freq='M')  # Predict for next 6 months
    future_data = pd.DataFrame({'Date': future_dates})
    future_data['Year'] = future_data['Date'].dt.year
    future_data['Month'] = future_data['Date'].dt.month

    # Assign mean values from training data to the future data
    future_data['Inflation'] = X_train['Inflation'].mean()
    future_data['Unemployment'] = X_train['Unemployment'].mean()
    future_data['Temperature'] = X_train['Temperature'].mean()

    # Ensure that the columns in future_X are in the same order as in X_train
    future_X = future_data[['Year', 'Month', 'Inflation', 'Unemployment', 'Temperature']]

    # Predict sales for future dates
    future_sales_predictions = linear_regressor.predict(future_X)

    # DPI and size settings for the plots
    dpi = 70
    fig_width = width / dpi
    fig_height = height / dpi

    # Plot Actual vs Predicted Sales
    fig1, ax1 = plt.subplots(figsize=(fig_width, fig_height))
    ax1.set_facecolor("#2E2E2E")
    ax1.plot(dataframe['Date'], y, label='Actual Sales', color='white')
    ax1.plot(future_dates, future_sales_predictions, label='Predicted Sales', linestyle='--', color='red')
    format_plot(ax1, 'Actual vs Predicted Sales', x_labels_freq)

    # Plot Prediction Accuracy (True vs Predicted with regression line and confidence interval)
    fig2 = plt.figure(figsize=(fig_width, fig_height))
    ax2 = sns.regplot(x=y_pred, y=y_test, scatter=True, fit_reg=True, color='white')
    format_plot(ax2, 'Prediction Accuracy', x_labels_freq)
    ax2.collections[1].set_edgecolor("#2E2E2E")  # Set confidence interval line color to match background

    return fig1, fig2

def use_decision_tree_regressor(csv_file_path, width=800, height=600, x_labels_freq=10):
    # Load and preprocess the data
    dataframe = pd.read_csv(csv_file_path)
    dataframe['Date'] = pd.to_datetime(dataframe['Date'])

    # Feature Engineering: Extract year and month from 'Date'
    dataframe['Year'] = dataframe['Date'].dt.year
    dataframe['Month'] = dataframe['Date'].dt.month

    # Define the features (X) and the target (y)
    X = dataframe[['Year', 'Month', 'Inflation', 'Unemployment', 'Temperature']]  # Example features
    y = dataframe['Sales']

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Linear Regression model
    decision_tree_regressor = DecisionTreeRegressor(random_state=42)
    decision_tree_regressor.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = decision_tree_regressor.predict(X_test)

    # Predict future sales
    future_dates = pd.date_range(start=dataframe['Date'].max() + pd.DateOffset(1), periods=18, freq='M')  # Predict for next 6 months
    future_data = pd.DataFrame({'Date': future_dates})
    future_data['Year'] = future_data['Date'].dt.year
    future_data['Month'] = future_data['Date'].dt.month

    # Assign mean values from training data to the future data
    future_data['Inflation'] = X_train['Inflation'].mean()
    future_data['Unemployment'] = X_train['Unemployment'].mean()
    future_data['Temperature'] = X_train['Temperature'].mean()

    # Ensure that the columns in future_X are in the same order as in X_train
    future_X = future_data[['Year', 'Month', 'Inflation', 'Unemployment', 'Temperature']]

    # Predict sales for future dates
    future_sales_predictions = decision_tree_regressor.predict(future_X)

    # DPI and size settings for the plots
    dpi = 70
    fig_width = width / dpi
    fig_height = height / dpi

    # Plot Actual vs Predicted Sales
    fig1, ax1 = plt.subplots(figsize=(fig_width, fig_height))
    ax1.set_facecolor("#2E2E2E")
    ax1.plot(dataframe['Date'], y, label='Actual Sales', color='white')
    ax1.plot(future_dates, future_sales_predictions, label='Predicted Sales', linestyle='--', color='red')
    format_plot(ax1, 'Actual vs Predicted Sales', x_labels_freq)

    # Plot Prediction Accuracy (True vs Predicted with regression line and confidence interval)
    fig2 = plt.figure(figsize=(fig_width, fig_height))
    ax2 = sns.regplot(x=y_pred, y=y_test, scatter=True, fit_reg=True, color='white')
    format_plot(ax2, 'Prediction Accuracy', x_labels_freq)
    ax2.collections[1].set_edgecolor("#2E2E2E")  # Set confidence interval line color to match background

    return fig1, fig2

def use_random_forest_regressor(csv_file_path, width=800, height=600, x_labels_freq=10):
    # Load and preprocess the data
    dataframe = pd.read_csv(csv_file_path)
    dataframe['Date'] = pd.to_datetime(dataframe['Date'])

    # Feature Engineering: Extract year and month from 'Date'
    dataframe['Year'] = dataframe['Date'].dt.year
    dataframe['Month'] = dataframe['Date'].dt.month

    # Define the features (X) and the target (y)
    X = dataframe[['Year', 'Month', 'Inflation', 'Unemployment', 'Temperature']]  # Example features
    y = dataframe['Sales']

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Linear Regression model
    random_forest_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    random_forest_regressor.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = random_forest_regressor.predict(X_test)

    # Predict future sales
    future_dates = pd.date_range(start=dataframe['Date'].max() + pd.DateOffset(1), periods=18, freq='M')  # Predict for next 6 months
    future_data = pd.DataFrame({'Date': future_dates})
    future_data['Year'] = future_data['Date'].dt.year
    future_data['Month'] = future_data['Date'].dt.month

    # Assign mean values from training data to the future data
    future_data['Inflation'] = X_train['Inflation'].mean()
    future_data['Unemployment'] = X_train['Unemployment'].mean()
    future_data['Temperature'] = X_train['Temperature'].mean()

    # Ensure that the columns in future_X are in the same order as in X_train
    future_X = future_data[['Year', 'Month', 'Inflation', 'Unemployment', 'Temperature']]

    # Predict sales for future dates
    future_sales_predictions = random_forest_regressor.predict(future_X)

    # DPI and size settings for the plots
    dpi = 70
    fig_width = width / dpi
    fig_height = height / dpi

    # Plot Actual vs Predicted Sales
    fig1, ax1 = plt.subplots(figsize=(fig_width, fig_height))
    ax1.set_facecolor("#2E2E2E")
    ax1.plot(dataframe['Date'], y, label='Actual Sales', color='white')
    ax1.plot(future_dates, future_sales_predictions, label='Predicted Sales', linestyle='--', color='red')
    format_plot(ax1, 'Actual vs Predicted Sales', x_labels_freq)

    # Plot Prediction Accuracy (True vs Predicted with regression line and confidence interval)
    fig2 = plt.figure(figsize=(fig_width, fig_height))
    ax2 = sns.regplot(x=y_pred, y=y_test, scatter=True, fit_reg=True, color='white')
    format_plot(ax2, 'Prediction Accuracy', x_labels_freq)
    ax2.collections[1].set_edgecolor("#2E2E2E")  # Set confidence interval line color to match background

    return fig1, fig2