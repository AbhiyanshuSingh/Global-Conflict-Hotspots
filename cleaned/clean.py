
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

# Extract unique country names from each dataset
sipri_countries = set(sipri_milex_data['Country'].unique())
conflict_countries = set(conflict_data['Country'].unique())
development_countries = set(development_data['Country'].unique())
arms_countries = set(arms_data['Country'].unique())
refugee_origin_countries = set(refugee_origin_data['Country'].unique())
refugee_asylum_countries = set(refugee_asylum_data['Country'].unique())
polity_countries = set(polity_data['Country'].unique())

# Compare country counts
print(f"SIPRI dataset: {len(sipri_countries)} countries")
print(f"Conflict dataset: {len(conflict_countries)} countries")
print(f"Development dataset: {len(development_countries)} countries")
print(f"Arms dataset: {len(arms_countries)} countries")
print(f"Refugee origin dataset: {len(refugee_origin_countries)} countries")
print(f"Refugee asylum dataset: {len(refugee_asylum_countries)} countries")
print(f"Polity dataset: {len(polity_countries)} countries")

# Create a standard country name dictionary
# Start with countries from the conflict dataset as the base
standard_country_names = {country: country for country in conflict_countries}

# Function to find closest matching country name
def find_closest_match(name, reference_list):
    if name in reference_list:
        return name
    matches = difflib.get_close_matches(name, reference_list, n=1, cutoff=0.8)
    return matches[0] if matches else None

# Create mapping dictionaries for each dataset
sipri_mapping = {country: find_closest_match(country, conflict_countries) for country in sipri_countries}
development_country_map = {country: find_closest_match(country, conflict_countries) 
                           for country in development_countries}
arms_country_map = {country: find_closest_match(country, conflict_countries) 
                    for country in arms_countries}
refugee_origin_country_map = {country: find_closest_match(country, conflict_countries) 
                              for country in refugee_origin_countries}
refugee_asylum_country_map = {country: find_closest_match(country, conflict_countries) 
                              for country in refugee_asylum_countries}
polity_country_map = {country: find_closest_match(country, conflict_countries) 
                      for country in polity_countries}

# Identify unmatched countries
unmapped_sipri = set(sipri_countries) - set(conflict_countries)
unmapped_development = [c for c, m in development_country_map.items() if m is None]
unmapped_arms = [c for c, m in arms_country_map.items() if m is None]
unmapped_refugee_origin = [c for c, m in refugee_origin_country_map.items() if m is None]
unmapped_refugee_asylum = [c for c, m in refugee_asylum_country_map.items() if m is None]
unmapped_polity = [c for c, m in polity_country_map.items() if m is None]

print("\nCountries without matches in SIPRI data:", len(unmapped_sipri))
print("\nCountries without matches in development data:", len(unmapped_development))
print("Countries without matches in arms data:", len(unmapped_arms))
print("Countries without matches in refugee origin data:", len(unmapped_refugee_origin))
print("Countries without matches in refugee asylum data:", len(unmapped_refugee_asylum))
print("Countries without matches in polity data:", len(unmapped_polity))

# Save mapping reports for manual review if needed
unmapped_df = pd.concat([
    pd.DataFrame({
        'Original': unmapped_development,
        'Dataset': ['Development'] * len(unmapped_development)
    }),
    pd.DataFrame({
        'Original': unmapped_sipri,
        'Dataset': ['SIPRI'] * len(unmapped_sipri)
    }),
    pd.DataFrame({
        'Original': unmapped_arms,
        'Dataset': ['Arms'] * len(unmapped_arms)
    }),
    pd.DataFrame({
        'Original': unmapped_refugee_origin,
        'Dataset': ['Refugee Origin'] * len(unmapped_refugee_origin)
    }),
    pd.DataFrame({
        'Original': unmapped_refugee_asylum,
        'Dataset': ['Refugee Asylum'] * len(unmapped_refugee_asylum)
    }),
    pd.DataFrame({
        'Original': unmapped_polity,
        'Dataset': ['Polity'] * len(unmapped_polity)
    })
])

unmapped_df.to_csv("country_name_issues.csv", index=False)