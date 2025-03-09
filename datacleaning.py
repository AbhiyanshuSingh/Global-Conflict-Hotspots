# import pandas as pd
# import numpy as np
# from pathlib import Path



# def clean_milex_sheet(df, sheet_name, fill_method='none'):
#     """
#     Clean a SIPRI MILEX dataset sheet with enhanced missing value handling.
    
#     Parameters:
#     df (pandas.DataFrame): Input DataFrame with countries as rows and years as columns
#     sheet_name (str): Name of the sheet being processed
#     fill_method (str): Method to handle missing values
    
#     Returns:
#     pandas.DataFrame: Cleaned DataFrame in long format
#     """
#     # Create a copy to avoid modifying the original
#     cleaned_df = df.copy()
    
#     # Melt the DataFrame to convert from wide to long format
#     melted_df = cleaned_df.melt(
#         id_vars=['Country'],
#         var_name='Year',
#         value_name='Value'
#     )
    
#     # Clean country names
#     melted_df['Country'] = melted_df['Country'].str.strip()
    
#     # Convert Year to integer
#     melted_df['Year'] = pd.to_numeric(melted_df['Year'], errors='coerce')
    
#     # Create a missing value indicator column
#     melted_df['Missing_Indicator'] = 'Available'
    
#     # Handle missing values and create indicators
#     missing_values = ['...', 'xxx', '.', '', ' ']
#     for mv in missing_values:
#         mask = melted_df['Value'].astype(str).str.contains(f'^{mv}$', na=False)
#         melted_df.loc[mask, 'Missing_Indicator'] = mv
    
#     # Convert values to float, setting missing values to NaN
#     melted_df['Value'] = pd.to_numeric(melted_df['Value'], errors='coerce')
    
    
#     # Add metadata columns
#     melted_df['Sheet_Name'] = sheet_name
#     melted_df['Metric_Type'] = get_metric_type(sheet_name)
    
#     # Sort by Country and Year
#     melted_df = melted_df.sort_values(['Country', 'Year'])
    
#     # Reset index
#     melted_df = melted_df.reset_index(drop=True)
    
#     return melted_df

# def get_metric_type(sheet_name):
#     """
#     Determine the metric type based on the sheet name.
#     """
#     metric_types = {
#         'US$': 'Currency_USD',
#         'local currency': 'Currency_Local',
#         '% of GDP': 'Percentage_GDP',
#         'per capita': 'Per_Capita',
#         '% of government': 'Percentage_Gov'
#     }
    
#     for key, value in metric_types.items():
#         if key.lower() in sheet_name.lower():
#             return value
#     return 'Other'

# def process_all_sheets(excel_path, fill_method='none', output_dir=None):
#     """
#     Process all sheets in the SIPRI MILEX Excel file.
    
#     Parameters:
#     excel_path (str): Path to the Excel file
#     fill_method (str): Method to handle missing values
#     output_dir (str): Directory to save individual CSV files (optional)
    
#     Returns:
#     dict: Dictionary containing cleaned DataFrames and summary reports
#     """
#     # Read all sheets
#     excel_file = pd.ExcelFile(excel_path)
#     sheets = excel_file.sheet_names
    
#     results = {
#         'cleaned_data': {},
#         'missing_data_reports': {},
#         'summary': {}
#     }
    
#     for sheet in sheets:
#         try:
#             # Read the sheet
#             df = pd.read_excel(excel_file, sheet_name=sheet)
            
#             # Clean the data
#             cleaned_df = clean_milex_sheet(df, sheet, fill_method)
            
#             # Generate missing data report
#             missing_report = analyze_missing_patterns(cleaned_df)
            
#             # Store results
#             results['cleaned_data'][sheet] = cleaned_df
#             results['missing_data_reports'][sheet] = missing_report
            
#             # Save to CSV if output directory is specified
#             if output_dir:
#                 output_path = Path(output_dir) / f"cleaned_{sheet.replace(' ', '_')}.csv"
#                 cleaned_df.to_csv(output_path, index=False)
            
#         except Exception as e:
#             print(f"Error processing sheet '{sheet}': {str(e)}")
    
#     # Generate overall summary
#     results['summary'] = generate_overall_summary(results)
    
#     return results

# def analyze_missing_patterns(df):
#     """
#     Analyze patterns in missing data.
#     """
#     return {
#         'missing_by_type': df['Missing_Indicator'].value_counts().to_dict(),
#         'missing_by_country': df[df['Missing_Indicator'] != 'Available'].groupby('Country').size().to_dict(),
#         'missing_by_year': df[df['Missing_Indicator'] != 'Available'].groupby('Year').size().to_dict(),
#         'total_records': len(df),
#         'missing_percentage': (len(df[df['Missing_Indicator'] != 'Available']) / len(df)) * 100
#     }

# def generate_overall_summary(results):
#     """
#     Generate overall summary of the data processing.
#     """
#     summary = {
#         'total_sheets_processed': len(results['cleaned_data']),
#         'total_records_processed': sum(len(df) for df in results['cleaned_data'].values()),
#         'missing_data_by_sheet': {
#             sheet: report['missing_percentage']
#             for sheet, report in results['missing_data_reports'].items()
#         }
#     }
#     return summary

# def save_summary_report(results, output_path):
#     """
#     Save a detailed summary report to a text file.
#     """
#     with open(output_path, 'w') as f:
#         f.write("SIPRI MILEX Data Processing Summary\n")
#         f.write("=" * 50 + "\n\n")
        
#         # Overall statistics
#         f.write("Overall Statistics:\n")
#         f.write(f"Total sheets processed: {results['summary']['total_sheets_processed']}\n")
#         f.write(f"Total records processed: {results['summary']['total_records_processed']}\n\n")
        
#         # Per-sheet statistics
#         f.write("Missing Data by Sheet:\n")
#         for sheet, percentage in results['summary']['missing_data_by_sheet'].items():
#             f.write(f"{sheet}: {percentage:.2f}% missing\n")


# # Set up paths
# excel_path = "D:\datavisual-claude\SIPRI-Milex-data-1948-2023 (1).xlsx"
# output_dir = "D:\datavisual-claude"  # Where you want to save the cleaned CSV files

# # Process all sheets
# results = process_all_sheets(
#     excel_path=excel_path,
#     fill_method='none',  # or 'interpolate', 'forward', 'backward'
#     output_dir=output_dir
# )

# # Save summary report
# save_summary_report(results, f"{output_dir}/processing_summary.txt")

# # Access cleaned data for specific sheets
# usd_data = results['cleaned_data']['Mil. exp. by country in US$']
# gdp_data = results['cleaned_data']['Mil. exp. by country as % of GDP']


#################################################----------------------------------############################################


# import pandas as pd
# import numpy as np
# import glob
# import os

# def load_and_prepare_sheet(file_path):
#     """
#     Load a single SIPRI sheet and prepare it for merging
#     """
#     # Read the CSV file
#     df = pd.read_csv(file_path)
    
#     # Convert '...' to NaN
#     df['Value'] = df['Value'].replace('...', np.nan)
    
#     # Convert to numeric
#     df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    
#     # Extract metric type from file name or Sheet_Name
#     metric_type = df['Sheet_Name'].iloc[0]
    
#     return df, metric_type

# def merge_sipri_sheets(folder_path):
#     """
#     Merge all SIPRI sheets into a single dataframe
#     """
#     # Get all CSV files in the folder
#     csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    
#     # Dictionary to store dataframes
#     sheet_data = {}
    
#     # Load each sheet
#     for file in csv_files:
#         df, metric_type = load_and_prepare_sheet(file)
        
#         # Create a pivot table for each sheet
#         pivot_df = df.pivot(
#             index=['Country', 'Year'],
#             columns='Sheet_Name',
#             values='Value'
#         ).reset_index()
        
#         # Store in dictionary
#         sheet_data[metric_type] = pivot_df
    
#     # Merge all sheets based on Country and Year
#     final_df = None
#     for metric_type, df in sheet_data.items():
#         if final_df is None:
#             final_df = df
#         else:
#             final_df = final_df.merge(
#                 df,
#                 on=['Country', 'Year'],
#                 how='outer'
#             )
    
#     # Clean up column names
#     final_df.columns = [col.replace(' ', '_').lower() for col in final_df.columns]
    
#     # Add year-over-year changes for each metric
#     value_columns = [col for col in final_df.columns 
#                     if col not in ['country', 'year']]
    
#     for col in value_columns:
#         final_df[f'{col}_change'] = final_df.groupby('country')[col].pct_change()
        
#         # Add 5-year rolling average
#         final_df[f'{col}_5yr_avg'] = final_df.groupby('country')[col].transform(
#             lambda x: x.rolling(window=5, min_periods=1).mean()
#         )
    
#     return final_df

# def validate_merged_data(df):
#     """
#     Validate the merged dataset
#     """
#     print("\nMerged Data Summary:")
#     print("-------------------")
#     print(f"Number of countries: {df['country'].nunique()}")
#     print(f"Year range: {df['year'].min()} - {df['year'].max()}")
    
#     print("\nColumns in merged dataset:")
#     for col in df.columns:
#         missing = df[col].isna().sum()
#         print(f"{col}: {missing} missing values ({(missing/len(df))*100:.2f}%)")
    
#     return None

# def main():
#     # Replace with your folder path containing SIPRI CSV files
#     folder_path = 'D:\datavisual-claude\sipri-milex-cleaned'
    
#     # Merge all sheets
#     merged_df = merge_sipri_sheets(folder_path)
    
#     # Validate the merged data
#     validate_merged_data(merged_df)
    
#     # Save the merged dataset
#     merged_df.to_csv('sipri_merged_final.csv', index=False)
    
#     # Print sample of merged data
#     print("\nSample of Merged Data:")
#     print(merged_df.head())

# if __name__ == "__main__":
#     main()



####################################----------UCDP-------------------############################################

# import pandas as pd
# import numpy as np
# from datetime import datetime

# def clean_ucdp_data(df):
#     """
#     Clean and process UCDP conflict data
#     """
#     # Create a copy of the dataframe
#     df_cleaned = df.copy()
    
#     # Keep only necessary columns
#     columns_to_keep = [
#         'location','side_a','side_b', 'year', 'intensity_level', 'type_of_conflict',
#         'region', 'incompatibility', 'start_date', 'end_date'
#     ]
#     df_cleaned = df_cleaned[columns_to_keep]
    
#     # Create mappings for categorical variables
#     type_of_conflict_map = {
#         1: 'extrasystemic',
#         2: 'interstate',
#         3: 'internal',
#         4: 'internationalized_internal'
#     }
    
#     region_map = {
#         1: 'europe',
#         2: 'middle_east',
#         3: 'asia',
#         4: 'africa',
#         5: 'americas'
#     }
    
#     incompatibility_map = {
#         1: 'territory',
#         2: 'government',
#         3: 'both'
#     }
    
#     # Apply mappings
#     df_cleaned['type_of_conflict'] = df_cleaned['type_of_conflict'].map(type_of_conflict_map)
#     df_cleaned['region'] = df_cleaned['region'].map(region_map)
#     df_cleaned['incompatibility'] = df_cleaned['incompatibility'].map(incompatibility_map)
    
#     # Convert dates to datetime
#     df_cleaned['start_date'] = pd.to_datetime(df_cleaned['start_date'])
#     df_cleaned['end_date'] = pd.to_datetime(df_cleaned['end_date'])
    
#     # Calculate conflict duration in days
#     df_cleaned['conflict_duration'] = (df_cleaned['end_date'] - df_cleaned['start_date']).dt.days
    
#     # Create summary statistics by country and year
#     summary = df_cleaned.groupby(['location', 'year']).agg({
#         'intensity_level': ['count', 'mean', 'max'],
#         'conflict_duration': ['mean', 'sum'],
#         'type_of_conflict': lambda x: x.nunique(),
#         'incompatibility': lambda x: x.value_counts().to_dict()
#     }).reset_index()
    
#     # Flatten column names
#     summary.columns = [
#         'country', 'year', 
#         'num_conflicts', 'avg_intensity', 'max_intensity',
#         'avg_duration', 'total_duration', 'conflict_type_count',
#         'incompatibility_types'
#     ]
    
#     # Create binary columns for conflict types
#     for conflict_type in incompatibility_map.values():
#         summary[f'has_{conflict_type}_conflict'] = summary['incompatibility_types'].apply(
#             lambda x: 1 if conflict_type in x else 0
#         )
    
#     # Drop the dictionary column
#     summary = summary.drop('incompatibility_types', axis=1)
    
#     # Add rolling statistics
#     summary['conflicts_past_3years'] = summary.groupby('country')['num_conflicts'].rolling(
#         window=3, min_periods=1
#     ).mean().reset_index(0, drop=True)
    
#     summary['intensity_past_3years'] = summary.groupby('country')['avg_intensity'].rolling(
#         window=3, min_periods=1
#     ).mean().reset_index(0, drop=True)
    
#     return summary

# def validate_cleaned_data(df):
#     """
#     Validate the cleaned dataset and print summary statistics
#     """
#     print("\nCleaned Data Summary:")
#     print("--------------------")
#     print(f"Number of countries: {df['country'].nunique()}")
#     print(f"Year range: {df['year'].min()} - {df['year'].max()}")
    
#     print("\nConflict Statistics:")
#     print(f"Total conflicts recorded: {df['num_conflicts'].sum()}")
#     print(f"Average conflicts per country-year: {df['num_conflicts'].mean():.2f}")
#     print(f"Maximum conflicts in a single country-year: {df['num_conflicts'].max()}")
    
#     print("\nMissing Values:")
#     print(df.isnull().sum())
    
#     return None

# def main():
#     # Read the UCDP data
#     file_path = r'D:\datavisual-claude\UcdpPrioConflict_v24_1.csv'  # Replace with your file path
#     df = pd.read_csv(file_path)
    
#     # Clean the data
#     cleaned_df = clean_ucdp_data(df)
    
#     # Validate the cleaning
#     validate_cleaned_data(cleaned_df)
    
#     # Save the cleaned data
#     cleaned_df.to_csv('ucdp_cleaned.csv', index=False)
    
#     # Print sample of cleaned data
#     print("\nSample of Cleaned Data:")
#     print(cleaned_df.head())

# if __name__ == "__main__":
#     main()



####################################--------Deveolopment------------######################################



# import pandas as pd
# import numpy as np

# # Load the development dataset
# # Adjust the file path as needed
# dev_data = pd.read_csv(r"D:\datavisual-claude\worldbankdevelopment.csv",encoding='ISO-8859-1')

# # Step 1: Reshape from wide to long format
# # First, identify the year columns
# year_columns = [col for col in dev_data.columns if col.startswith('19') or col.startswith('20')]

# # Melt the dataframe to convert from wide to long format
# dev_data_long = pd.melt(
#     dev_data,
#     id_vars=['Country Name', 'Country Code', 'Series Name', 'Series Code'],
#     value_vars=year_columns,
#     var_name='Year Column',
#     value_name='Value'
# )

# # Step 2: Clean the Year column - extract just the year
# dev_data_long['Year'] = dev_data_long['Year Column'].str.extract(r'(\d{4})').astype(int)
# dev_data_long.drop('Year Column', axis=1, inplace=True)

# # Step 3: Clean the Value column - handle non-numeric values
# # Replace '..' and other non-numeric placeholders with NaN
# dev_data_long['Value'] = pd.to_numeric(dev_data_long['Value'], errors='coerce')

# # Step 4: Now we can safely pivot
# dev_data_wide = dev_data_long.pivot_table(
#     index=['Country Name', 'Country Code', 'Year'],
#     columns='Series Name',
#     values='Value',
#     aggfunc='first'  # Use 'first' to avoid aggregation issues
# ).reset_index()

# # Step 5: Clean column names
# dev_data_wide.columns.name = None
# dev_data_wide.rename(columns={'Country Name': 'Country'}, inplace=True)

# # Step 6: Save the cleaned dataset with missing values preserved
# dev_data_wide.to_csv("development_data_cleaned.csv", index=False)

# # Display sample of the cleaned data
# print(dev_data_wide.head())
# print(f"Cleaned dataset shape: {dev_data_wide.shape}")
# print(f"Missing values count: {dev_data_wide.isna().sum().sum()}")
# print(f"Missing values by column:")
# for col in dev_data_wide.columns:
#     missing = dev_data_wide[col].isna().sum()
#     if missing > 0:
#         print(f"  {col}: {missing} missing values ({missing/len(dev_data_wide)*100:.1f}%)")



# ################################---------Import-Export-Arms-----------#######################################

# import pandas as pd
# import numpy as np

# # Load the arms import/export dataset
# # Adjust the file path as needed
# arms_data = pd.read_csv(r"D:\datavisual-claude\import-export-values_1950-2023.csv")

# # Step 1: Identify which columns to keep and which to convert
# # We'll keep recipient country and individual year columns
# year_columns = [str(year) for year in range(1950, 2024)]  # Years from 1950 to 2023
# id_columns = ['Recipient']

# # Validate that the columns exist in the dataset
# available_year_columns = [col for col in year_columns if col in arms_data.columns]

# # Step 2: Reshape from wide to long format
# arms_data_long = pd.melt(
#     arms_data,
#     id_vars=id_columns,
#     value_vars=available_year_columns,
#     var_name='Year',
#     value_name='Arms_Transfer_Value'
# )

# # Step 3: Convert Year column to numeric
# arms_data_long['Year'] = pd.to_numeric(arms_data_long['Year'])

# # Step 4: Handle special values according to dataset documentation
# # - 0 means between 0 and 0.5 million
# # - NaN means no trade happened
# # First convert to numeric, which will turn non-numeric values to NaN
# arms_data_long['Arms_Transfer_Value'] = pd.to_numeric(arms_data_long['Arms_Transfer_Value'], errors='coerce')

# # Step 5: Add a trade status column for analysis clarity
# arms_data_long['Trade_Status'] = 'Active Trade'
# arms_data_long.loc[arms_data_long['Arms_Transfer_Value'] == 0, 'Trade_Status'] = 'Minimal (0-0.5M)'
# arms_data_long.loc[arms_data_long['Arms_Transfer_Value'].isna(), 'Trade_Status'] = 'No Trade'

# # Step 6: Rename columns for consistency with other datasets
# arms_data_long.rename(columns={'recipient': 'Country'}, inplace=True)

# # Step 7: Save the cleaned dataset
# arms_data_long.to_csv("arms_transfer_cleaned.csv", index=False)

# # Display sample of the cleaned data
# print(arms_data_long.head())
# print(f"Cleaned dataset shape: {arms_data_long.shape}")
# print("Value distribution:")
# print(arms_data_long['Trade_Status'].value_counts())



##################-------------------Refugees--------------------######################################


# import pandas as pd
# import numpy as np

# # Load the refugee dataset
# # Adjust the file path as needed
# refugee_data = pd.read_csv(r"D:\datavisual-claude\refugee.csv")

# # Step 1: Check for any non-numeric values in the metric columns
# metric_columns = [
#     'Refugees under UNHCR\'s mandate', 
#     'Asylum-seekers',
#     'Returned refugees', 
#     'IDPs of concern to UNHCR', 
#     'Returned IDPss', 
#     'Stateless persons', 
#     'Others of concern', 
#     'Other people in need of international protection', 
#     'Host Community'
# ]

# # Step 2: Convert metric columns to numeric, handling any non-numeric values
# for col in metric_columns:
#     if col in refugee_data.columns:
#         refugee_data[col] = pd.to_numeric(refugee_data[col], errors='coerce')

# # Step 3: Create a total refugees/displaced persons column
# refugee_data['Total_Displaced'] = refugee_data[metric_columns].sum(axis=1)

# # Step 4: Rename columns for consistency
# refugee_data.rename(columns={
#     'Country of origin': 'Origin_Country',
#     'Country of asylum': 'Asylum_Country',
# }, inplace=True)

# # Step 5: Create country-focused datasets for different analyses

# # Origin country perspective (outflow of refugees)
# refugee_by_origin = refugee_data.groupby(['Origin_Country', 'Year'])[metric_columns + ['Total_Displaced']].sum().reset_index()
# refugee_by_origin.rename(columns={'Origin_Country': 'Country'}, inplace=True)

# # Asylum country perspective (inflow of refugees)
# refugee_by_asylum = refugee_data.groupby(['Asylum_Country', 'Year'])[metric_columns + ['Total_Displaced']].sum().reset_index()
# refugee_by_asylum.rename(columns={'Asylum_Country': 'Country'}, inplace=True)

# # Step 6: Save the cleaned datasets
# refugee_data.to_csv("refugee_data_full_cleaned.csv", index=False)
# refugee_by_origin.to_csv("refugee_data_by_origin_cleaned.csv", index=False)
# refugee_by_asylum.to_csv("refugee_data_by_asylum_cleaned.csv", index=False)

# # Display samples and stats
# print("Full refugee dataset sample:")
# print(refugee_data.head())
# print(f"Full dataset shape: {refugee_data.shape}")

# print("\nRefugee by origin country sample:")
# print(refugee_by_origin.head())
# print(f"Origin dataset shape: {refugee_by_origin.shape}")

# print("\nRefugee by asylum country sample:")
# print(refugee_by_asylum.head())
# print(f"Asylum dataset shape: {refugee_by_asylum.shape}")

# print("\nMissing values by column:")
# for col in refugee_data.columns:
#     missing = refugee_data[col].isna().sum()
#     if missing > 0:
#         print(f"  {col}: {missing} missing values ({missing/len(refugee_data)*100:.1f}%)")




###########################--------------Polity----------##########################################

# import pandas as pd
# import numpy as np

# # Load the polity dataset
# # Adjust the file path as needed
# polity_data = pd.read_csv(r"D:\datavisual-claude\polity.csv")

# # Step 1: Select the most relevant columns
# # These are the core columns that will be most useful for conflict analysis
# relevant_columns = [
#     'country', 'year',  # Identifiers
#     'democ', 'autoc', 'polity', 'polity2',  # Core regime metrics
#     'durable',  # Regime durability (years since last substantive change)
#     'regtrans'  # Regime transition indicator
# ]

# # Filter to keep only relevant columns
# polity_clean = polity_data[relevant_columns].copy()

# # Step 2: Rename columns for consistency and clarity
# polity_clean.rename(columns={
#     'country': 'Country',
#     'year': 'Year',
#     'democ': 'Democracy_Score',  # Democracy score (0-10)
#     'autoc': 'Autocracy_Score',  # Autocracy score (0-10)
#     'polity': 'Polity_Score',    # Combined score (-10 to 10)
#     'polity2': 'Polity2_Score',  # Revised combined score (-10 to 10)
#     'durable': 'Regime_Durability_Years',
#     'regtrans': 'Regime_Transition_Type'
# }, inplace=True)

# # Step 3: Add a categorical regime type column for easier analysis
# def categorize_regime(row):
#     polity2 = row['Polity2_Score']
#     if pd.isna(polity2):
#         return "Unknown"
#     elif polity2 >= 6:
#         return "Democracy"
#     elif polity2 >= 1:
#         return "Open Anocracy"
#     elif polity2 >= -5:
#         return "Closed Anocracy"
#     else:  # -10 to -6
#         return "Autocracy"

# polity_clean['Regime_Type'] = polity_clean.apply(categorize_regime, axis=1)

# # Step 4: Convert any special codes to NaN
# # In Polity dataset, values like -66, -77, -88 represent special cases
# special_codes = [-66, -77, -88, -99]
# for col in ['Democracy_Score', 'Autocracy_Score', 'Polity_Score', 'Polity2_Score']:
#     polity_clean[col] = polity_clean[col].apply(lambda x: np.nan if x in special_codes else x)

# # Step 5: Create a political stability indicator
# # Higher numbers = more stable
# polity_clean['Political_Stability'] = polity_clean['Regime_Durability_Years'].copy()
# polity_clean.loc[polity_clean['Regime_Transition_Type'].notna(), 'Political_Stability'] *= 0.5

# # Step 6: Save the cleaned dataset
# polity_clean.to_csv("polity_data_cleaned.csv", index=False)

# # Display sample and stats
# print("Polity dataset sample:")
# print(polity_clean.head())
# print(f"Dataset shape: {polity_clean.shape}")

# print("\nRegime type distribution:")
# print(polity_clean['Regime_Type'].value_counts())

# print("\nMissing values by column:")
# for col in polity_clean.columns:
#     missing = polity_clean[col].isna().sum()
#     if missing > 0:
#         print(f"  {col}: {missing} missing values ({missing/len(polity_clean)*100:.1f}%)")



#################------------------- Merging them--------------######################################

import pandas as pd
import numpy as np
import difflib

# Load all cleaned datasets
sipri_milex_data = pd.read_csv(r"D:\datavisual-claude\cleaned\sipri_merged_final.csv")
conflict_data = pd.read_csv(r"D:\datavisual-claude\cleaned\ucdp_cleaned.csv")
development_data = pd.read_csv(r"D:\datavisual-claude\cleaned\development_data_cleaned.csv")
arms_data = pd.read_csv(r"D:\datavisual-claude\cleaned\arms_transfer_cleaned.csv")
refugee_origin_data = pd.read_csv(r"D:\datavisual-claude\cleaned\refugee_data_by_origin_cleaned.csv")
refugee_asylum_data = pd.read_csv(r"D:\datavisual-claude\cleaned\refugee_data_by_asylum_cleaned.csv")
polity_data = pd.read_csv(r"D:\datavisual-claude\cleaned\polity_data_cleaned.csv")

# First, standardize the column names - ensure the country column has the same name in all datasets
sipri_milex_data = sipri_milex_data.rename(columns={'Country': 'country'})
conflict_data = conflict_data.rename(columns={'Country': 'country'})
development_data = development_data.rename(columns={'Country': 'country'})
arms_data = arms_data.rename(columns={'Country': 'country'})
refugee_origin_data = refugee_origin_data.rename(columns={'Country': 'country'})
refugee_asylum_data = refugee_asylum_data.rename(columns={'Country': 'country'})
polity_data = polity_data.rename(columns={'Country': 'country'})

# Also standardize the year column names if needed
sipri_milex_data = sipri_milex_data.rename(columns={'Year': 'year'})
conflict_data = conflict_data.rename(columns={'Year': 'year'})
development_data = development_data.rename(columns={'Year': 'year'})
arms_data = arms_data.rename(columns={'Year': 'year'})
refugee_origin_data = refugee_origin_data.rename(columns={'Year': 'year'})
refugee_asylum_data = refugee_asylum_data.rename(columns={'Year': 'year'})
polity_data = polity_data.rename(columns={'Year': 'year'})

# Now perform the merges one by one
# Start with the conflict data as the base
merged_data = conflict_data.copy()

# Merge with SIPRI military expenditure data
merged_data = merged_data.merge(
    sipri_milex_data, 
    on=['country', 'year'], 
    how='outer',
    suffixes=('', '_sipri')
)

# Merge with development data
merged_data = merged_data.merge(
    development_data, 
    on=['country', 'year'], 
    how='outer',
    suffixes=('', '_dev')
)

# Merge with arms transfer data
merged_data = merged_data.merge(
    arms_data, 
    on=['country', 'year'], 
    how='outer',
    suffixes=('', '_arms')
)

# Merge with refugee origin data
merged_data = merged_data.merge(
    refugee_origin_data, 
    on=['country', 'year'], 
    how='outer',
    suffixes=('', '_refugee_origin')
)

# Merge with refugee asylum data
merged_data = merged_data.merge(
    refugee_asylum_data, 
    on=['country', 'year'], 
    how='outer',
    suffixes=('', '_refugee_asylum')
)

# Merge with polity data
merged_data = merged_data.merge(
    polity_data, 
    on=['country', 'year'], 
    how='outer',
    suffixes=('', '_polity')
)

# Check the merged dataset
print(f"Merged dataset shape: {merged_data.shape}")
print(f"Number of countries in merged dataset: {merged_data['country'].nunique()}")
print(f"Year range in merged dataset: {merged_data['year'].min()} to {merged_data['year'].max()}")

# Save the merged dataset
merged_data.to_csv(r"D:\datavisual-claude\cleaned\merged_conflict_dataset.csv", index=False)

# Optional: Generate a summary of missing values
missing_data_summary = pd.DataFrame({
    'Column': merged_data.columns,
    'Missing Values': merged_data.isnull().sum(),
    'Missing Percentage': (merged_data.isnull().sum() / len(merged_data) * 100).round(2)
})
missing_data_summary = missing_data_summary.sort_values('Missing Percentage', ascending=False)
missing_data_summary.to_csv(r"D:\datavisual-claude\cleaned\missing_data_summary.csv", index=False)