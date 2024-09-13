import sys
import streamlit as st
import tempfile
import os
import pandas as pd
import plotly.express as px
import numpy as np
import base64
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import json
import xlsxwriter
from io import BytesIO

# Set up Streamlit app title
st.title("CSViewer")

# Pagination settings
ROWS_PER_PAGE = 100  # Number of rows to display per page
MAX_ROWS = 50000  # Maximum number of rows allowed

# Function to handle CSV files
def process_file(file_path):
    try:
        # Read the first row to get column names
        df = pd.read_csv(file_path, nrows=1)
        columns = df.columns

        # Count the total number of rows
        total_rows = sum(1 for _ in open(file_path)) - 1  # Subtract 1 for header row

        if total_rows > MAX_ROWS:
            st.error(f"The uploaded file has {total_rows} rows, which exceeds the maximum limit of {MAX_ROWS} rows. Please upload a smaller file.")
            return None

        # If within limit, read the entire file
        df = pd.read_csv(file_path, low_memory=False)
        return df

    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

# Function to display a paginated view of the data
def display_paginated_data(df, total_rows):
    total_pages = (total_rows // ROWS_PER_PAGE) + 1
    page_number = st.number_input(f"Select page (1-{total_pages})", min_value=1, max_value=total_pages)
    
    start_row = (page_number - 1) * ROWS_PER_PAGE
    end_row = start_row + ROWS_PER_PAGE
    st.write(f"Displaying rows {start_row} to {end_row} out of {total_rows}")
    
    df_page = df.iloc[start_row:end_row]
    st.write(df_page)

# Function to display column filtering and visualization
def display_column_filter_and_visualization(df):
    # Column-based filtering
    st.subheader('Filter Data')
    columns = df.columns.tolist()
    selected_column = st.selectbox("Select Column to filter by", columns)
    
    unique_values = df[selected_column].unique()
    selected_value = st.selectbox("Select value", unique_values)
    
    filtered_df = df[df[selected_column] == selected_value]
    
    st.write("Filtered DataFrame:")
    st.write(filtered_df.head())

    return filtered_df

# Function to display overall data visualization
def display_overall_visualization(df):
    st.subheader("Overall Data Visualization")

    # Correlation heatmap for numerical columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 1:
        correlation_matrix = df[numeric_columns].corr()
        fig_heatmap = px.imshow(correlation_matrix, 
                                title="Correlation Heatmap of Numerical Columns",
                                labels=dict(color="Correlation"),
                                x=correlation_matrix.columns,
                                y=correlation_matrix.columns)
        st.plotly_chart(fig_heatmap)
    else:
        st.write("Not enough numerical columns for correlation heatmap.")

# Function to allow CSV download
def get_csv_download_link(df, filename="filtered_data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV file</a>'
    return href

# Function to handle missing values
def handle_missing_values(df):
    st.subheader("Handle Missing Values")
    missing_columns = df.columns[df.isnull().any()].tolist()
    if not missing_columns:
        st.write("No missing values found in the dataset.")
        return df
    
    for column in missing_columns:
        st.write(f"Column: {column}")
        strategy = st.selectbox(f"Choose strategy for {column}", 
                                ["Drop rows", "Fill with mean", "Fill with median", "Fill with mode", "Fill with custom value"])
        if strategy == "Drop rows":
            df = df.dropna(subset=[column])
        elif strategy == "Fill with mean":
            df[column].fillna(df[column].mean(), inplace=True)
        elif strategy == "Fill with median":
            df[column].fillna(df[column].median(), inplace=True)
        elif strategy == "Fill with mode":
            df[column].fillna(df[column].mode()[0], inplace=True)
        elif strategy == "Fill with custom value":
            custom_value = st.text_input(f"Enter custom value for {column}")
            df[column].fillna(custom_value, inplace=True)
    
    return df

# Function to remove duplicates
def remove_duplicates(df):
    st.subheader("Remove Duplicates")
    duplicate_count = df.duplicated().sum()
    st.write(f"Number of duplicate rows: {duplicate_count}")
    if duplicate_count > 0:
        if st.button("Remove duplicate rows"):
            df = df.drop_duplicates()
            st.success(f"{duplicate_count} duplicate rows removed.")
    return df

# Function to normalize data
def normalize_data(df):
    st.subheader("Normalize Data")
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) == 0:
        st.write("No numeric columns to normalize.")
        return df
    
    columns_to_normalize = st.multiselect("Select columns to normalize", numeric_columns)
    if columns_to_normalize:
        normalization_method = st.selectbox("Choose normalization method", ["Min-Max Scaling", "Z-Score Normalization"])
        if normalization_method == "Min-Max Scaling":
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        
        df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
        st.success(f"Normalized {len(columns_to_normalize)} columns using {normalization_method}.")
    
    return df

# Function to create additional charts
def create_additional_charts(df):
    st.subheader("Additional Charts")
    chart_type = st.selectbox("Select chart type", ["Bar Chart", "Line Chart", "Pie Chart"])
    
    if chart_type == "Bar Chart":
        x_column = st.selectbox("Select X-axis column", df.columns)
        y_column = st.selectbox("Select Y-axis column", df.select_dtypes(include=[np.number]).columns)
        fig = px.bar(df, x=x_column, y=y_column, title=f"Bar Chart: {x_column} vs {y_column}")
    
    elif chart_type == "Line Chart":
        x_column = st.selectbox("Select X-axis column", df.columns)
        y_column = st.selectbox("Select Y-axis column", df.select_dtypes(include=[np.number]).columns)
        fig = px.line(df, x=x_column, y=y_column, title=f"Line Chart: {x_column} vs {y_column}")
    
    elif chart_type == "Pie Chart":
        value_column = st.selectbox("Select value column", df.select_dtypes(include=[np.number]).columns)
        names_column = st.selectbox("Select names column", df.select_dtypes(exclude=[np.number]).columns)
        fig = px.pie(df, values=value_column, names=names_column, title=f"Pie Chart: {names_column} ({value_column})")
    
    st.plotly_chart(fig)

# Function to perform data transformations
def perform_data_transformations(df):
    st.subheader("Data Transformations")
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) == 0:
        st.write("No numeric columns to transform.")
        return df
    
    column_to_transform = st.selectbox("Select column to transform", numeric_columns)
    transformation = st.selectbox("Select transformation", ["Log", "Square Root", "Square", "Cube"])
    
    if transformation == "Log":
        df[f"{column_to_transform}_log"] = np.log(df[column_to_transform])
    elif transformation == "Square Root":
        df[f"{column_to_transform}_sqrt"] = np.sqrt(df[column_to_transform])
    elif transformation == "Square":
        df[f"{column_to_transform}_squared"] = df[column_to_transform] ** 2
    elif transformation == "Cube":
        df[f"{column_to_transform}_cubed"] = df[column_to_transform] ** 3
    
    st.success(f"Applied {transformation} transformation to {column_to_transform}.")
    return df

# Function to perform data aggregation
def perform_data_aggregation(df):
    st.subheader("Data Aggregation")
    
    # Select columns for grouping
    group_columns = st.multiselect("Select columns to group by", df.columns)
    
    if group_columns:
        # Select columns to aggregate
        agg_columns = st.multiselect("Select columns to aggregate", df.select_dtypes(include=[np.number]).columns)
        
        if agg_columns:
            # Select aggregation functions
            agg_functions = st.multiselect("Select aggregation functions", ["Mean", "Sum", "Count", "Min", "Max"])
            
            if agg_functions:
                agg_dict = {col: [func.lower() for func in agg_functions] for col in agg_columns}
                aggregated_df = df.groupby(group_columns).agg(agg_dict)
                st.write("Aggregated Data:")
                st.write(aggregated_df)
                
                # Provide download link for aggregated data
                st.markdown(get_csv_download_link(aggregated_df, "aggregated_data.csv"), unsafe_allow_html=True)
        else:
            st.write("Please select columns to aggregate.")
    else:
        st.write("Please select columns to group by.")

# Function to detect outliers
def detect_outliers(df):
    st.subheader("Outlier Detection")
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_columns) == 0:
        st.write("No numeric columns to detect outliers.")
        return
    
    column = st.selectbox("Select column for outlier detection", numeric_columns)
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    st.write(f"Number of outliers detected: {len(outliers)}")
    st.write("Outliers:")
    st.write(outliers)
    
    fig = px.box(df, y=column, title=f"Box Plot of {column}")
    st.plotly_chart(fig)

# Function to export data in different formats
def export_data(df):
    st.subheader("Export Data")
    export_format = st.selectbox("Select export format", ["CSV", "Excel", "JSON"])
    
    if export_format == "CSV":
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="exported_data.csv">Download CSV file</a>'
    elif export_format == "Excel":
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        b64 = base64.b64encode(output.getvalue()).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="exported_data.xlsx">Download Excel file</a>'
    else:  # JSON
        json_str = df.to_json(orient='records')
        b64 = base64.b64encode(json_str.encode()).decode()
        href = f'<a href="data:application/json;base64,{b64}" download="exported_data.json">Download JSON file</a>'
    
    st.markdown(href, unsafe_allow_html=True)

# Function to generate data profile
def generate_data_profile(df):
    st.subheader("Data Profile")
    
    # Overall info
    st.write("Dataset Overview:")
    st.write(f"Number of rows: {df.shape[0]}")
    st.write(f"Number of columns: {df.shape[1]}")
    
    # Column-wise info
    st.write("\nColumn-wise Information:")
    for column in df.columns:
        st.write(f"\nColumn: {column}")
        st.write(f"Data type: {df[column].dtype}")
        st.write(f"Number of unique values: {df[column].nunique()}")
        st.write(f"Number of missing values: {df[column].isnull().sum()}")
        
        if df[column].dtype in ['int64', 'float64']:
            st.write(f"Min value: {df[column].min()}")
            st.write(f"Max value: {df[column].max()}")
            st.write(f"Mean value: {df[column].mean():.2f}")
            st.write(f"Median value: {df[column].median()}")
        
        if df[column].dtype == 'object':
            st.write("Most frequent values:")
            st.write(df[column].value_counts().head())

# Function to perform correlation analysis
def perform_correlation_analysis(df):
    st.subheader("Correlation Analysis")
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) < 2:
        st.write("Not enough numeric columns for correlation analysis.")
        return
    
    selected_columns = st.multiselect("Select columns for correlation analysis", numeric_columns)
    
    if len(selected_columns) < 2:
        st.write("Please select at least two columns for correlation analysis.")
        return
    
    correlation_matrix = df[selected_columns].corr()
    
    # Heatmap
    fig_heatmap = px.imshow(correlation_matrix, 
                            title="Correlation Heatmap",
                            labels=dict(color="Correlation"),
                            x=correlation_matrix.columns,
                            y=correlation_matrix.columns)
    st.plotly_chart(fig_heatmap)
    
    # Scatter plot matrix
    fig_scatter = px.scatter_matrix(df[selected_columns], 
                                    title="Scatter Plot Matrix",
                                    labels={col:col for col in selected_columns})
    st.plotly_chart(fig_scatter)

# File uploader and processing logic
upload_file = st.file_uploader("Choose a CSV file", type=["csv"])

if upload_file is not None:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        tmp_file.write(upload_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        df = process_file(tmp_file_path)

        if df is not None:
            st.success("File processed successfully.")
            
            # Data preview
            st.subheader("Data Preview")
            st.write(df.head())

            # Data cleaning options
            df = handle_missing_values(df)
            df = remove_duplicates(df)
            df = normalize_data(df)

            # Data transformations
            df = perform_data_transformations(df)

            # Display paginated data
            total_rows = len(df)
            display_paginated_data(df, total_rows)

            # Display overall data visualization
            display_overall_visualization(df)

            # Create additional charts
            create_additional_charts(df)

            # Perform data aggregation
            perform_data_aggregation(df)

            # Detect outliers
            detect_outliers(df)

            # Display column filtering and visualization
            filtered_df = display_column_filter_and_visualization(df)

            # New features
            export_data(df)
            generate_data_profile(df)
            perform_correlation_analysis(df)

    except Exception as e:
        st.error(f"An error occurred: {e}")

    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
else:
    st.info("Please upload a CSV file.")
