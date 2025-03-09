import joblib
import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import StandardScaler
import country_converter as coco
import traceback
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from datetime import datetime, timedelta

# Create a dictionary of country coordinates
COUNTRY_COORDINATES = {
    'Afghanistan': {'lat': 33.93911, 'lon': 67.709953},
    'Albania': {'lat': 41.153332, 'lon': 20.168331},
    'Algeria': {'lat': 28.033886, 'lon': 1.659626},
    'Angola': {'lat': -11.202692, 'lon': 17.873887},
    'Argentina': {'lat': -38.416097, 'lon': -63.616672},
    'Armenia': {'lat': 40.069099, 'lon': 45.038189},
    'Australia': {'lat': -25.274398, 'lon': 133.775136},
    'Austria': {'lat': 47.516231, 'lon': 14.550072},
    'Azerbaijan': {'lat': 40.143105, 'lon': 47.576927},
    'Bangladesh': {'lat': 23.684994, 'lon': 90.356331},
    'Belarus': {'lat': 53.709807, 'lon': 27.953389},
    'Belgium': {'lat': 50.503887, 'lon': 4.469936},
    'Benin': {'lat': 9.30769, 'lon': 2.315834},
    'Bhutan': {'lat': 27.514162, 'lon': 90.433601},
    'Bolivia': {'lat': -16.290154, 'lon': -63.588653},
    'Bosnia and Herzegovina': {'lat': 43.915886, 'lon': 17.679076},
    'Botswana': {'lat': -22.328474, 'lon': 24.684866},
    'Brazil': {'lat': -14.235004, 'lon': -51.92528},
    'Bulgaria': {'lat': 42.733883, 'lon': 25.48583},
    'Burkina Faso': {'lat': 12.238333, 'lon': -1.561593},
    'Burma': {'lat': 21.916221, 'lon': 95.955974},
    'Burundi': {'lat': -3.373056, 'lon': 29.918886},
    'Cambodia': {'lat': 12.565679, 'lon': 104.990963},
    'Cameroon': {'lat': 7.369722, 'lon': 12.354722},
    'Canada': {'lat': 56.130366, 'lon': -106.346771},
    'Central African Republic': {'lat': 6.611111, 'lon': 20.939444},
    'Chad': {'lat': 15.454166, 'lon': 18.732207},
    'Chile': {'lat': -35.675147, 'lon': -71.542969},
    'China': {'lat': 35.86166, 'lon': 104.195397},
    'Colombia': {'lat': 4.570868, 'lon': -74.297333},
    'Congo': {'lat': -0.228021, 'lon': 15.827659},
    'Costa Rica': {'lat': 9.748917, 'lon': -83.753428},
    'Croatia': {'lat': 45.1, 'lon': 15.2},
    'Cuba': {'lat': 21.521757, 'lon': -77.781167},
    'Cyprus': {'lat': 35.126413, 'lon': 33.429859},
    'Czech Republic': {'lat': 49.817492, 'lon': 15.472962},
    'Denmark': {'lat': 56.26392, 'lon': 9.501785},
    'Djibouti': {'lat': 11.825138, 'lon': 42.590275},
    'Dominican Republic': {'lat': 18.735693, 'lon': -70.162651},
    'DR Congo': {'lat': -4.038333, 'lon': 21.758664},
    'Ecuador': {'lat': -1.831239, 'lon': -78.183406},
    'Egypt': {'lat': 26.820553, 'lon': 30.802498},
    'El Salvador': {'lat': 13.794185, 'lon': -88.89653},
    'Eritrea': {'lat': 15.179384, 'lon': 39.782334},
    'Estonia': {'lat': 58.595272, 'lon': 25.013607},
    'Ethiopia': {'lat': 9.145, 'lon': 40.489673},
    'Finland': {'lat': 61.92411, 'lon': 25.748151},
    'France': {'lat': 46.227638, 'lon': 2.213749},
    'Gabon': {'lat': -0.803689, 'lon': 11.609444},
    'Georgia': {'lat': 42.315407, 'lon': 43.356892},
    'Germany': {'lat': 51.165691, 'lon': 10.451526},
    'Ghana': {'lat': 7.946527, 'lon': -1.023194},
    'Greece': {'lat': 39.074208, 'lon': 21.824312},
    'Guatemala': {'lat': 15.783471, 'lon': -90.230759},
    'Guinea': {'lat': 9.945587, 'lon': -9.696645},
    'Haiti': {'lat': 18.971187, 'lon': -72.285215},
    'Honduras': {'lat': 15.199999, 'lon': -86.241905},
    'Hungary': {'lat': 47.162494, 'lon': 19.503304},
    'India': {'lat': 20.593684, 'lon': 78.96288},
    'Indonesia': {'lat': -0.789275, 'lon': 113.921327},
    'Iran': {'lat': 32.427908, 'lon': 53.688046},
    'Iraq': {'lat': 33.223191, 'lon': 43.679291},
    'Ireland': {'lat': 53.41291, 'lon': -8.24389},
    'Israel': {'lat': 31.046051, 'lon': 34.851612},
    'Italy': {'lat': 41.87194, 'lon': 12.56738},
    'Jamaica': {'lat': 18.109581, 'lon': -77.297508},
    'Japan': {'lat': 36.204824, 'lon': 138.252924},
    'Jordan': {'lat': 30.585164, 'lon': 36.238414},
    'Kazakhstan': {'lat': 48.019573, 'lon': 66.923684},
    'Kenya': {'lat': -0.023559, 'lon': 37.906193},
    'Kosovo': {'lat': 42.602636, 'lon': 20.902977},
    'Kuwait': {'lat': 29.31166, 'lon': 47.481766},
    'Kyrgyzstan': {'lat': 41.20438, 'lon': 74.766098},
    'Laos': {'lat': 19.85627, 'lon': 102.495496},
    'Latvia': {'lat': 56.879635, 'lon': 24.603189},
    'Lebanon': {'lat': 33.854721, 'lon': 35.862285},
    'Lesotho': {'lat': -29.609988, 'lon': 28.233608},
    'Liberia': {'lat': 6.428055, 'lon': -9.429499},
    'Libya': {'lat': 26.3351, 'lon': 17.228331},
    'Lithuania': {'lat': 55.169438, 'lon': 23.881275},
    'Luxembourg': {'lat': 49.815273, 'lon': 6.129583},
    'Madagascar': {'lat': -18.766947, 'lon': 46.869107},
    'Malawi': {'lat': -13.254308, 'lon': 34.301525},
    'Malaysia': {'lat': 4.210484, 'lon': 101.975766},
    'Mali': {'lat': 17.570692, 'lon': -3.996166},
    'Mexico': {'lat': 23.634501, 'lon': -102.552784},
    'Moldova': {'lat': 47.411631, 'lon': 28.369885},
    'Mongolia': {'lat': 46.862496, 'lon': 103.846656},
    'Montenegro': {'lat': 42.708678, 'lon': 19.37439},
    'Morocco': {'lat': 31.791702, 'lon': -7.09262},
    'Mozambique': {'lat': -18.665695, 'lon': 35.529562},
    'Myanmar': {'lat': 21.916221, 'lon': 95.955974},
    'Namibia': {'lat': -22.95764, 'lon': 18.49041},
    'Nepal': {'lat': 28.394857, 'lon': 84.124008},
    'Netherlands': {'lat': 52.132633, 'lon': 5.291266},
    'New Zealand': {'lat': -40.900557, 'lon': 174.885971},
    'Nicaragua': {'lat': 12.865416, 'lon': -85.207229},
    'Niger': {'lat': 17.607789, 'lon': 8.081666},
    'Nigeria': {'lat': 9.081999, 'lon': 8.675277},
    'North Korea': {'lat': 40.339852, 'lon': 127.510093},
    'North Macedonia': {'lat': 41.608635, 'lon': 21.745275},
    'Norway': {'lat': 60.472024, 'lon': 8.468946},
    'Oman': {'lat': 21.512583, 'lon': 55.923255},
    'Pakistan': {'lat': 30.375321, 'lon': 69.345116},
    'Palestine': {'lat': 31.952162, 'lon': 35.233154},
    'Panama': {'lat': 8.537981, 'lon': -80.782127},
    'Papua New Guinea': {'lat': -6.314993, 'lon': 143.95555},
    'Paraguay': {'lat': -23.442503, 'lon': -58.443832},
    'Peru': {'lat': -9.189967, 'lon': -75.015152},
    'Philippines': {'lat': 12.879721, 'lon': 121.774017},
    'Poland': {'lat': 51.919438, 'lon': 19.145136},
    'Portugal': {'lat': 39.399872, 'lon': -8.224454},
    'Qatar': {'lat': 25.354826, 'lon': 51.183884},
    'Romania': {'lat': 45.943161, 'lon': 24.96676},
    'Russia': {'lat': 61.52401, 'lon': 105.318756},
    'Rwanda': {'lat': -1.940278, 'lon': 29.873888},
    'Saudi Arabia': {'lat': 23.885942, 'lon': 45.079162},
    'Senegal': {'lat': 14.497401, 'lon': -14.452362},
    'Serbia': {'lat': 44.016521, 'lon': 21.005859},
    'Sierra Leone': {'lat': 8.460555, 'lon': -11.779889},
    'Singapore': {'lat': 1.352083, 'lon': 103.819836},
    'Slovakia': {'lat': 48.669026, 'lon': 19.699024},
    'Slovenia': {'lat': 46.151241, 'lon': 14.995463},
    'Somalia': {'lat': 5.152149, 'lon': 46.199616},
    'South Africa': {'lat': -30.559482, 'lon': 22.937506},
    'South Korea': {'lat': 35.907757, 'lon': 127.766922},
    'South Sudan': {'lat': 6.876991, 'lon': 31.306978},
    'Spain': {'lat': 40.463667, 'lon': -3.74922},
    'Sri Lanka': {'lat': 7.873054, 'lon': 80.771797},
    'Sudan': {'lat': 12.862807, 'lon': 30.217636},
    'Sweden': {'lat': 60.128161, 'lon': 18.643501},
    'Switzerland': {'lat': 46.818188, 'lon': 8.227512},
    'Syria': {'lat': 34.802075, 'lon': 38.996815},
    'Taiwan': {'lat': 23.69781, 'lon': 120.960515},
    'Tajikistan': {'lat': 38.861034, 'lon': 71.276093},
    'Tanzania': {'lat': -6.369028, 'lon': 34.888822},
    'Thailand': {'lat': 15.870032, 'lon': 100.992541},
    'Togo': {'lat': 8.619543, 'lon': 0.824782},
    'Tunisia': {'lat': 33.886917, 'lon': 9.537499},
    'Turkey': {'lat': 38.963745, 'lon': 35.243322},
    'Turkmenistan': {'lat': 38.969719, 'lon': 59.556278},
    'Uganda': {'lat': 1.373333, 'lon': 32.290275},
    'Ukraine': {'lat': 48.379433, 'lon': 31.16558},
    'United Arab Emirates': {'lat': 23.424076, 'lon': 53.847818},
    'United Kingdom': {'lat': 55.378051, 'lon': -3.435973},
    'United States of America': {'lat': 37.09024, 'lon': -95.712891},
    'Uruguay': {'lat': -32.522779, 'lon': -55.765835},
    'Uzbekistan': {'lat': 41.377491, 'lon': 64.585262},
    'Venezuela': {'lat': 6.42375, 'lon': -66.58973},
    'Vietnam': {'lat': 14.058324, 'lon': 108.277199},
    'Yemen': {'lat': 15.552727, 'lon': 48.516388},
    'Zambia': {'lat': -13.133897, 'lon': 27.849332},
    'Zimbabwe': {'lat': -19.015438, 'lon': 29.154857}
}

# Set page config
st.set_page_config(layout="wide", page_title="Global Conflict Hotspots")

@st.cache_data
def load_and_prepare_data(file_path='your_data.csv'):
    try:
        # Load your data
        df = pd.read_csv(file_path)
        
        # Create a dataframe from the coordinates dictionary
        coords_df = pd.DataFrame.from_dict(COUNTRY_COORDINATES, orient='index').reset_index()
        coords_df.columns = ['country', 'latitude', 'longitude']
        
        # Merge with your original data
        df = df.merge(coords_df, on='country', how='left')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.write("Please check if your CSV file contains the required columns and is properly formatted.")
        return None


# def make_predictions(df):
#     """
#     Make conflict intensity predictions with a simplified model
#     """
#     try:
#         # Select only the most essential features
#         essential_features = [
#             'num_conflicts', 'avg_intensity', 'max_intensity', 'avg_duration', 
#             'total_duration', 'conflict_type_count', 'has_territory_conflict',
#             'has_government_conflict', 'has_both_conflict', 
#             'conflicts_past_3years', 'intensity_past3years'
#         ]
        
#         # Political and stability indicators
#         political_features = [
#             'Democracy_Score', 'Autocracy_Score', 'Polity_Score', 'Polity2_Score',
#             'Regime_Durability_Years', 'Political_Stability'
#         ]

#         # Economic indicators
#         economic_features = [
#             'GDP per capita (current US$)', 'GDP per capita, PPP (current international $)',
#             'Total natural resources rents (% of GDP)', 
#             'Unemployment, total (% of total labor force) (national estimate)'
#         ]

#         # Displacement and refugee indicators
#         displacement_features = [
#             'Refugees under UNHCR\'s mandate', 'Asylum-seekers', 'IDPs of concern to UNHCR',
#             'Total_Displaced'
#         ]

#         # Military indicators
#         military_features = [
#             'Arms_Transfer_Value', 'constant(2022)_us$', 'share_of_gdp', 'share_of_govt.spending'
#         ]

#         # Combine all feature categories
#         all_potential_features = (
#             essential_features + political_features + 
#             economic_features + displacement_features + 
#             military_features
#         )

#         # Filter to only use features that exist in the dataset
#         features = [f for f in all_potential_features if f in df.columns]
        
#         if not features:
#             st.error("No usable features found in the dataset")
#             return None
            
#         # Prepare features
#         X = df[features].copy()
        
#         # Simple imputation with mean values
#         for col in X.columns:
#             X[col] = X[col].fillna(X[col].mean())
        
#         # Use standard scaler
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(X)
        
#         # Select target variable
#         if 'max_intensity' in df.columns:
#             y = df['max_intensity'].fillna(0)
#         elif 'avg_intensity' in df.columns:
#             y = df['avg_intensity'].fillna(0)
#         else:
#             # Create a synthetic target from available conflict indicators
#             conflict_indicators = [col for col in ['conflicts_past_3years', 'intensity_past3years', 
#                                                   'num_conflicts', 'conflict_type_count'] 
#                                   if col in df.columns]
            
#             if conflict_indicators:
#                 y = df[conflict_indicators].mean(axis=1)
#             else:
#                 st.warning("No direct conflict indicators found, model may be less accurate")
#                 # Use political stability as a proxy if available
#                 if 'Political_Stability' in df.columns:
#                     y = 10 - df['Political_Stability']  # Invert so lower stability = higher conflict risk
#                 else:
#                     # Create a placeholder target
#                     y = pd.Series(5, index=df.index)  # Default mid-range risk
        
#         # Simplified Random Forest with fewer trees
#         model = RandomForestRegressor(
#             n_estimators=100,  # Fewer trees
#             max_depth=8,      # Limit depth
#             min_samples_leaf=5, # Prevent overfitting
#             random_state=42   # For reproducibility
#         )
        
#         # Fit model
#         model.fit(X_scaled, y)
        
#         # Make predictions
#         predictions = model.predict(X_scaled)
        
#         # Scale predictions to 0-10 range
#         if len(predictions) > 0 and predictions.max() > predictions.min():
#             predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min()) * 10
#         else:
#             predictions = np.ones_like(predictions) * 5
        
#         # Add predictions to the dataframe
#         df['predicted_intensity'] = predictions
        
        
#         return df
        
#     except Exception as e:
#         st.error(f"Error in prediction: {str(e)}")
#         return None


def train_predictive_model(df):
    """
    Train the predictive model and return predictions
    """
    features = [
        'num_conflicts', 'avg_intensity', 'max_intensity', 'avg_duration', 
        'total_duration', 'conflict_type_count', 'has_territory_conflict',
        'has_government_conflict', 'has_both_conflict', 
        'conflicts_past_3years', 'intensity_past3years',
        'Democracy_Score', 'Autocracy_Score', 'Polity_Score', 'Polity2_Score',
        'Regime_Durability_Years', 'Political_Stability',
        'GDP per capita (current US$)', 'GDP per capita, PPP (current international $)',
        'Total natural resources rents (% of GDP)', 'Unemployment, total (% of total labor force)',
        'Refugees under UNHCR\'s mandate', 'Asylum-seekers', 'IDPs of concern to UNHCR',
        'Total_Displaced', 'Arms_Transfer_Value'
    ]

    features = [f for f in features if f in df.columns]  # Keep only available columns

    #old
    # target = 'max_intensity' if 'max_intensity' in df.columns else 'avg_intensity'
    #new
    conflict_indicators = ['max_intensity', 'avg_intensity', 'conflicts_past_3years', 'intensity_past3years', 'num_conflicts']
    # Ensure these columns exist in the dataset
    valid_indicators = [col for col in conflict_indicators if col in df.columns]

    if valid_indicators:
        #old
        # df['conflict_risk'] = df[valid_indicators].sum(axis=1)  # Sum multiple conflict-related factors
        #new
        # Assign different weights to more relevant features
        df['conflict_risk'] = (
            df.get('max_intensity', 0) * 2 +  # More weight to direct conflict intensity
            df.get('avg_intensity', 0) * 2 +  
            df.get('conflicts_past_3years', 0) * 3 +  # More weight to recent conflicts
            df.get('intensity_past3years', 0) * 3 +
            df.get('num_conflicts', 0) * 3 +
            df.get('Political_Stability', 0) * -1 +  # Lower stability = higher risk
            df.get('GDP per capita (current US$)', 0) * -0.5  # Less importance to GDP
        )
        df['conflict_risk'] = (df['conflict_risk'] - df['conflict_risk'].min()) / (df['conflict_risk'].max() - df['conflict_risk'].min()) * 10

    else:
        df['conflict_risk'] = 1  # Default to 1 if no conflict data is available

    target = 'conflict_risk'


    df[target] = df[target].fillna(0)

    #new
    df[target] = (df[target] - df[target].min()) / (df[target].max() - df[target].min()) * 10


    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #old
    # model = RandomForestRegressor(n_estimators=150, max_depth=10, min_samples_leaf=5, random_state=42)
    #new
    model = RandomForestRegressor(
    n_estimators=500,  # More trees for deeper learning
    max_depth=20,  # Increase depth to learn complex patterns
    min_samples_split=3,  # Allow more splits
    min_samples_leaf=2,  # Prevent overfitting
    random_state=42
    )


    # #new
    # feature_importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    # st.write("Feature Importance:", feature_importances)



    model.fit(X_train_scaled, y_train)

    joblib.dump(model, "conflict_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    df['predicted_intensity'] = model.predict(scaler.transform(df[features]))

    return df[['country', 'latitude', 'longitude', 'predicted_intensity']]


def aggregate_predictions(df):
    """
    Aggregate predictions to ensure one intensity per country
    """
    df = df.groupby('country', as_index=False).agg({
        'latitude': 'first',
        'longitude': 'first',
        'predicted_intensity': 'mean'  # Use average intensity per country
    })
    return df



def clean_metric_value(value):
    """Clean and format metric values"""
    if pd.isna(value):
        return "N/A"
    if isinstance(value, (int, float)):
        if value > 1000000:  # Format millions
            return f"{value/1000000:.1f}M"
        if value > 1000:  # Format thousands
            return f"{value/1000:.1f}K"
        return f"{value:.2f}"
    return str(value)


# def create_globe_visualization(df):
#     """
#     Create simplified globe visualization
#     """
#     try:

#         # # Aggregate by country
#         # if 'country' in df.columns and 'predicted_intensity' in df.columns:
#         #     # For countries with multiple entries, take the most recent year if available
#         #     if 'year' in df.columns:
#         #         # Get the most recent year for each country
#         #         idx = df.groupby('country')['year'].idxmax()
#         #         map_data = df.loc[idx]
#         #     else:
#         #         # If no year column, aggregate by taking the maximum risk score
#         #         map_data = df.groupby('country').agg({
#         #             'latitude': 'first',
#         #             'longitude': 'first',
#         #             'predicted_intensity': 'max',
#         #         }).reset_index()
#         # else:
#         #     map_data = df.copy()

#         # Prepare data for visualization
#         map_data = df.copy()
        
        
#         # Ensure required columns exist
#         required_columns = ['country', 'latitude', 'longitude', 'predicted_intensity']
#         missing_columns = [col for col in required_columns if col not in map_data.columns]
        
#         if missing_columns:
#             st.error(f"Missing required columns for visualization: {', '.join(missing_columns)}")
#             return None
        
#         # Create color array directly
#         # Green (low risk) to red (high risk)
#         def get_color(intensity):
#             normalized = min(max(intensity / 10, 0), 1)
#             r = int(255 * normalized)
#             g = int(255 * (1 - normalized))
#             b = 0
#             return [r, g, 50]  # Adding some blue for better visibility
        
#         # Calculate colors 
#         map_data['color'] = map_data['predicted_intensity'].apply(get_color)
        
#         # # Simple radius calculation
#         # map_data['radius'] = map_data['predicted_intensity'] * 30000 + 20000
#         #new code
#         map_data['radius'] = (map_data['predicted_intensity'] * 2) * 40000 + 25000

        
#         # Create main visualization layer - just one layer for simplicity
#         circle_layer = pdk.Layer(
#             "ScatterplotLayer",
#             map_data,
#             get_position=["longitude", "latitude"],
#             get_radius="radius",
#             get_fill_color="color",
#             pickable=True,
#             opacity=0.6,
#             stroked=False,  # Remove stroke for performance
#             filled=True,
#         )

#         # Set the viewport location
#         view_state = pdk.ViewState(
#             latitude=20,
#             longitude=0,
#             zoom=1,
#             pitch=0,  # Flat view for performance
#         )

#         # Simple tooltip
#         tooltip = {
#             "html": "<b>{country}</b><br>Risk Score: {predicted_intensity}"
#         }

#         # Create and return the deck
#         deck = pdk.Deck(
#             layers=[circle_layer],
#             initial_view_state=view_state,
#             tooltip=tooltip,
#             map_style="mapbox://styles/mapbox/light-v10",
#         )

#         # # Create the layer
#         # layer = pdk.Layer(
#         #     "ScatterplotLayer",
#         #     map_data,
#         #     get_position=["longitude", "latitude"],
#         #     get_radius="predicted_intensity * 75000",  # Scale for visibility
#         #     get_fill_color=[
#         #         "255 * (predicted_intensity / 10)",  # Red component increases with risk
#         #         "100 * (1 - predicted_intensity / 10)",  # Green decreases with risk
#         #         "50 * (1 - predicted_intensity / 10)",  # Blue decreases with risk
#         #         "180"  # Alpha (transparency)
#         #     ],
#         #     pickable=True,
#         #     opacity=0.8,
#         #     stroked=True,
#         #     filled=True,
#         #     radiusMinPixels=5,
#         #     radiusMaxPixels=50,
#         # )

#         # # Set the viewport location
#         # view_state = pdk.ViewState(
#         #     latitude=20,
#         #     longitude=0,
#         #     zoom=1.5,
#         #     pitch=0,
#         # )

#         # # Combine everything and render
#         # deck = pdk.Deck(
#         #     layers=[layer],
#         #     initial_view_state=view_state,
#         #     tooltip={"text": "{country}\nRisk Score: {predicted_intensity}"},
#         # )

#         return deck

#     except Exception as e:
#         st.error(f"Error creating visualization: {str(e)}")
#         return None



def create_globe_visualization(df):
    """
    Create a world map visualization of predicted conflict hotspots
    """
    def get_color(intensity):
        normalized = min(max(intensity / 10, 0), 1)
        r = int(255 * normalized)
        g = int(255 * (1 - normalized))
        return [r, g, 50]

    df['color'] = df['predicted_intensity'].apply(get_color)
    df['radius'] = (df['predicted_intensity'] + 1) ** 2 * 50000  # Ensure visibility

    circle_layer = pdk.Layer(
        "ScatterplotLayer",
        df,
        get_position=["longitude", "latitude"],
        get_radius="radius",
        get_fill_color="color",
        pickable=True,
        opacity=0.6,
        stroked=False,
        filled=True,
    )

    view_state = pdk.ViewState(latitude=20, longitude=0, zoom=1, pitch=0)

    tooltip = {"html": "<b>{country}</b><br>Risk Score: {predicted_intensity}"}

    return pdk.Deck(layers=[circle_layer], initial_view_state=view_state, tooltip=tooltip, map_style="mapbox://styles/mapbox/light-v10")



def create_country_page(df, selected_country):
    """Create detailed analysis page for selected country"""
    
    # Get country specific data
    country_data = df[df['country'] == selected_country].copy()

    if len(country_data) == 0:
        st.error(f"No data available for {selected_country}")
        return
    
    st.header(f"📊 Detailed Analysis: {selected_country}")
    
    # Create two columns for the first row of visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # GDP and GDP per capita trends
        st.subheader("Economic Indicators Over Time")
        try:
            economic_data = pd.DataFrame({
                'Year': country_data['year'],
                'GDP per capita': country_data['GDP per capita (current US$)'],
                'GDP Growth US$': country_data['constant_(2022)_us$']
            }).dropna()
            
            if not economic_data.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=economic_data['Year'],
                    y=economic_data['GDP per capita'],
                    name='GDP per capita',
                    line=dict(color='blue')
                ))
                fig.add_trace(go.Scatter(
                    x=economic_data['Year'],
                    y=economic_data['GDP Growth US$'],
                    name='GDP Growth US$',
                    line=dict(color='green'),
                    yaxis='y2'
                ))
                
                fig.update_layout(
                    title='GDP Trends',
                    yaxis=dict(title='GDP per capita (USD)'),
                    yaxis2=dict(title='GDP Growth US$', overlaying='y', side='right'),
                    hovermode='x unified',
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No economic data available")
        except Exception as e:
            st.warning(f"Could not display economic indicators: {str(e)}")

    with col2:
        # Military Expenditure and Arms Trade
        st.subheader("Military Indicators")
        try:
            military_data = pd.DataFrame({
                'Year': country_data['year'],
                'Military Expenditure': country_data['share_of_gdp'],
                'Arms Trade': country_data['Arms_Transfer_Value']
            }).dropna()
            
            if not military_data.empty:            
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=military_data['Year'],
                    y=military_data['Military Expenditure'],
                    name='Military Expenditure',
                    marker_color='navy'
                ))
                fig.add_trace(go.Scatter(
                    x=military_data['Year'],
                    y=military_data['Arms Trade'],
                    name='Arms Trade',
                    line=dict(color='red'),
                    yaxis='y2'
                ))
                
                fig.update_layout(
                    title='Military Spending & Arms Trade',
                    yaxis=dict(title='% of GDP'),
                    yaxis2=dict(title='Trade in $millions', overlaying='y', side='right'),
                    hovermode='x unified',
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No military data available")
        except Exception as e:
            st.warning(f"Could not display military indicators: {str(e)}")

    # Create two columns for the second row of visualizations
    col3, col4 = st.columns(2)

    with col3:
        # Social Indicators
        st.subheader("Social Indicators")
        try:
            social_data = pd.DataFrame({
                'Year': country_data['year'],
                'Unemployment': country_data['Unemployment, total (% of total labor force) (national estimate)'],
                'Education Expenditure': country_data['Adjusted savings: education expenditure (current US$)']
            }).dropna()
            
            if not social_data.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=social_data['Year'],
                    y=social_data['Unemployment'],
                    name='Unemployment',
                    line=dict(color='orange')
                ))
                fig.add_trace(go.Scatter(
                    x=social_data['Year'],
                    y=social_data['Education Expenditure'],
                    name='Education Expenditure',
                    line=dict(color='purple'),
                    yaxis='y2'
                ))
                
                fig.update_layout(
                    title='Social Development Indicators',
                    yaxis=dict(title='Unemployment %'),
                    hovermode='x unified',
                    yaxis2=dict(title='$Billions', overlaying='y', side='right'),
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No social data available")
        except Exception as e:
            st.warning(f"Could not display social indicators: {str(e)}")

    with col4:
        # Stability and Democracy Scores
        st.subheader("Political Indicators")
        try:
            political_data = pd.DataFrame({
                'Year': country_data['year'],
                'Political Stability': country_data['Political_Stability'],
                'Democracy Score': country_data['Democracy_Score']
            }).dropna()
            
            if not political_data.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=political_data['Year'],
                    y=political_data['Political Stability'],
                    name='Political Stability',
                    line=dict(color='teal')
                ))
                fig.add_trace(go.Scatter(
                    x=political_data['Year'],
                    y=political_data['Democracy Score'],
                    name='Democracy Score',
                    line=dict(color='brown'),
                    yaxis='y2'
                ))
                
                fig.update_layout(
                    title='Political Stability & Democracy',
                    yaxis=dict(title='Polity Score'),
                    hovermode='x unified',
                    yaxis2=dict(title='Democracy Score', overlaying='y', side='right'),
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No political data available")
        except Exception as e:
            st.warning(f"Could not display political indicators: {str(e)}")

    # Create two columns for the second row of visualizations
    col5, col6 = st.columns(2)


    with col5:
        # Historical Conflict Trends
        st.subheader("Historical Conflict Trends")
        try:
            conflict_data = pd.DataFrame({
                'Year': country_data['year'],
                'Number of Conflicts': country_data['num_conflicts'],
                'Average Intensity': country_data['avg_intensity']
            }).dropna()
            
            if not conflict_data.empty:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=conflict_data['Year'],
                    y=conflict_data['Number of Conflicts'],
                    name='Number of Conflicts',
                    marker_color='firebrick'
                ))
                fig.add_trace(go.Scatter(
                    x=conflict_data['Year'],
                    y=conflict_data['Average Intensity'],
                    name='Average Intensity',
                    line=dict(color='yellow'),
                    yaxis='y2'
                ))
                
                fig.update_layout(
                    title='Historical Conflict Metrics',
                    yaxis=dict(title='Number of Conflicts'),
                    yaxis2=dict(title='Intensity Scale', overlaying='y', side='right'),
                    hovermode='x unified',
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No historical conflict data available")
        except Exception as e:
            st.warning(f"Could not display conflict trends: {str(e)}")

    with col6:
        # Conflict Duration Analysis
        st.subheader("Conflict Duration Analysis")
        try:
            duration_data = pd.DataFrame({
                'Year': country_data['year'],
                'Average Duration': country_data['avg_duration'],
                'Total Duration': country_data['total_duration']
            }).dropna()
            
            if not duration_data.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=duration_data['Year'],
                    y=duration_data['Average Duration'],
                    name='Average Duration',
                    line=dict(color='darkblue')
                ))
                fig.add_trace(go.Bar(
                    x=duration_data['Year'],
                    y=duration_data['Total Duration'],
                    name='Total Duration',
                    marker_color='royalblue',
                    yaxis='y2'
                ))
                
                fig.update_layout(
                    title='Conflict Duration Trends',
                    yaxis=dict(title='Average Duration (days)'),
                    yaxis2=dict(title='Total Duration (days)', overlaying='y', side='right'),
                    hovermode='x unified',
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No conflict duration data available")
        except Exception as e:
            st.warning(f"Could not display duration analysis: {str(e)}")

    # Add third row of new visualizations
    col7, col8 = st.columns(2)


    with col7:
        # Conflict Type Analysis
        st.subheader("Conflict Type Analysis")
        try:
            # Look for conflict type data columns
            conflict_type_columns = ['has_territory_conflict', 'has_government_conflict', 'has_both_conflict', 'num_conflicts', 'conflict_type_count']
            available_columns = [col for col in conflict_type_columns if col in country_data.columns]
            
            if len(available_columns) > 0:
                # Filter for year 2023 data first
                year_2023_data = country_data[country_data['year'] == 2023]
                
                # If we don't have 2023 data, get the most recent year available
                if year_2023_data.empty:
                    most_recent_year = country_data['year'].max()
                    recent_data = country_data[country_data['year'] == most_recent_year].iloc[0] if not country_data.empty else None
                    st.info(f"Data for 2023 not available. Showing data for {most_recent_year}.")
                else:
                    recent_data = year_2023_data.iloc[0]
                
                # Proceed only if we have data
                if recent_data is not None:
                    # Check if we have type-specific columns
                    if ('has_territory_conflict' in country_data.columns or 
                        'has_government_conflict' in country_data.columns or 
                        'has_both_conflict' in country_data.columns):
                        
                        # Create data for pie chart
                        labels = []
                        values = []
                        
                        # Try to get values, defaulting to 0 if columns don't exist or values are NaN
                        territory_val = 0
                        govt_val = 0
                        both_val = 0
                        
                        if 'has_territory_conflict' in country_data.columns:
                            territory_val = recent_data.get('has_territory_conflict', 0)
                            # Convert to numeric and handle NaN values
                            if pd.notna(territory_val):
                                # Handle different data types
                                if isinstance(territory_val, (bool, np.bool_)):
                                    territory_val = int(territory_val)
                                elif isinstance(territory_val, (int, float, np.number)):
                                    territory_val = int(territory_val)
                                else:
                                    try:
                                        territory_val = int(float(territory_val))
                                    except:
                                        territory_val = 0
                            else:
                                territory_val = 0
                        
                        if 'has_government_conflict' in country_data.columns:
                            govt_val = recent_data.get('has_government_conflict', 0)
                            # Convert to numeric and handle NaN values
                            if pd.notna(govt_val):
                                # Handle different data types
                                if isinstance(govt_val, (bool, np.bool_)):
                                    govt_val = int(govt_val)
                                elif isinstance(govt_val, (int, float, np.number)):
                                    govt_val = int(govt_val)
                                else:
                                    try:
                                        govt_val = int(float(govt_val))
                                    except:
                                        govt_val = 0
                            else:
                                govt_val = 0
                        
                        if 'has_both_conflict' in country_data.columns:
                            both_val = recent_data.get('has_both_conflict', 0)
                            # Convert to numeric and handle NaN values
                            if pd.notna(both_val):
                                # Handle different data types
                                if isinstance(both_val, (bool, np.bool_)):
                                    both_val = int(both_val)
                                elif isinstance(both_val, (int, float, np.number)):
                                    both_val = int(both_val)
                                else:
                                    try:
                                        both_val = int(float(both_val))
                                    except:
                                        both_val = 0
                            else:
                                both_val = 0
                        
                        # Add data for pie chart if values exist
                        if territory_val > 0:
                            labels.append('Territory')
                            values.append(territory_val)
                        
                        if govt_val > 0:
                            labels.append('Government')
                            values.append(govt_val)
                        
                        if both_val > 0:
                            labels.append('Both Types')
                            values.append(both_val)
                        
                        # If no specific types but we have num_conflicts
                        if len(labels) == 0 and 'num_conflicts' in country_data.columns:
                            num_val = recent_data.get('num_conflicts', 0)
                            if pd.notna(num_val) and num_val > 0:
                                labels = ['Unspecified']
                                values = [num_val]
                        
                        # Fallback to conflict_type_count if available
                        if len(labels) == 0 and 'conflict_type_count' in country_data.columns:
                            count_val = recent_data.get('conflict_type_count', 0)
                            if pd.notna(count_val) and count_val > 0:
                                labels = ['Unspecified']
                                values = [count_val]
                        
                        # If we still have no data, check if avg_intensity or max_intensity exists to infer conflicts
                        if len(labels) == 0:
                            if 'avg_intensity' in country_data.columns and pd.notna(recent_data.get('avg_intensity')):
                                intensity_val = recent_data.get('avg_intensity', 0)
                                if intensity_val > 0:
                                    labels = ['Active Conflicts']
                                    values = [1]  # At least one conflict
                            elif 'max_intensity' in country_data.columns and pd.notna(recent_data.get('max_intensity')):
                                intensity_val = recent_data.get('max_intensity', 0)
                                if intensity_val > 0:
                                    labels = ['Active Conflicts']
                                    values = [1]  # At least one conflict
                        
                        # Display data in a pie chart if we have data, otherwise show a message
                        if len(labels) > 0 and sum(values) > 0:
                            fig = go.Figure(data=[go.Pie(
                                labels=labels,
                                values=values,
                                hole=.3,
                                marker_colors=['crimson', 'navy', 'darkorange', 'gray']
                            )])
                            
                            year_display = int(recent_data['year']) if pd.notna(recent_data.get('year')) else "N/A"
                            fig.update_layout(
                                title=f'Types of Conflicts (Year: {year_display})',
                                annotations=[dict(text='Conflict<br>Types', x=0.5, y=0.5, font_size=12, showarrow=False)]
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            # No specific conflict data available, try to show time series
                            if 'num_conflicts' in country_data.columns:
                                conflict_years = country_data.dropna(subset=['num_conflicts'])
                                
                                if not conflict_years.empty:
                                    fig = go.Figure()
                                    
                                    fig.add_trace(go.Scatter(
                                        x=conflict_years['year'],
                                        y=conflict_years['num_conflicts'],
                                        mode='lines+markers',
                                        name='Number of Conflicts',
                                        line=dict(color='red', width=3),
                                        fill='tozeroy'
                                    ))
                                    
                                    fig.update_layout(
                                        title='Number of Conflicts Over Time',
                                        xaxis=dict(title='Year'),
                                        yaxis=dict(title='Number of Conflicts')
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("No conflict data available for this country")
                            else:
                                st.info("No conflict type data available for this country")
                    else:
                        # Fall back to time series of conflict numbers if available
                        if 'num_conflicts' in country_data.columns:
                            conflict_years = country_data.dropna(subset=['num_conflicts'])
                            
                            if not conflict_years.empty:
                                fig = go.Figure()
                                
                                fig.add_trace(go.Scatter(
                                    x=conflict_years['year'],
                                    y=conflict_years['num_conflicts'],
                                    mode='lines+markers',
                                    name='Number of Conflicts',
                                    line=dict(color='red', width=3),
                                    fill='tozeroy'
                                ))
                                
                                fig.update_layout(
                                    title='Number of Conflicts Over Time',
                                    xaxis=dict(title='Year'),
                                    yaxis=dict(title='Number of Conflicts')
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("No conflict data available for this country")
                        else:
                            st.info("No conflict data available for this country")
                else:
                    st.info("No data available for this country")
            else:
                st.info("No conflict-related columns found in the dataset")
        except Exception as e:
            st.warning(f"Could not display conflict type analysis: {str(e)}")
            # Print detailed traceback for debugging
            import traceback
            st.error(traceback.format_exc())


    with col8:
        # Displacement Analysis
        st.subheader("Displacement Impact")
        try:
            # Check if we have displacement data columns - use exact column names from your data file
            refugee_col = "Refugees under UNHCR's mandate"
            idp_col = 'IDPs of concern to UNHCR'
            asylum_col = 'Asylum-seekers'
            total_col = 'Total_Displaced'
            
            # Check if all required columns exist
            has_displacement_data = all(col in country_data.columns for col in [refugee_col, idp_col, asylum_col, total_col])
            
            if has_displacement_data:
                # Create a dataframe with the displacement data, handling missing values properly
                displacement_data = pd.DataFrame({
                    'Year': country_data['year'],
                    'Total Displaced': country_data[total_col].fillna(0),
                    'Refugees': country_data[refugee_col].fillna(0),
                    'IDPs': country_data[idp_col].fillna(0),
                    'Asylum Seekers': country_data[asylum_col].fillna(0)
                })
                
                # Remove rows where all displacement values are zero or NaN
                displacement_data = displacement_data[
                    (displacement_data['Total Displaced'] > 0) | 
                    (displacement_data['Refugees'] > 0) | 
                    (displacement_data['IDPs'] > 0) | 
                    (displacement_data['Asylum Seekers'] > 0)
                ]
                
                if not displacement_data.empty:
                    # Create a figure with secondary y-axis
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    # Add bar charts for individual displacement categories on primary y-axis
                    fig.add_trace(go.Bar(
                        x=displacement_data['Year'],
                        y=displacement_data['Refugees'],
                        name='Refugees',
                        marker_color='darkblue'
                    ), secondary_y=False)
                    
                    fig.add_trace(go.Bar(
                        x=displacement_data['Year'],
                        y=displacement_data['IDPs'],
                        name='IDPs',
                        marker_color='darkred'
                    ), secondary_y=False)
                    
                    fig.add_trace(go.Bar(
                        x=displacement_data['Year'],
                        y=displacement_data['Asylum Seekers'],
                        name='Asylum Seekers',
                        marker_color='darkorange'
                    ), secondary_y=False)
                    
                    # Add Total Displaced line on secondary y-axis
                    fig.add_trace(go.Scatter(
                        x=displacement_data['Year'],
                        y=displacement_data['Total Displaced'],
                        name='Total Displaced',
                        line=dict(color='black', width=3),
                        mode='lines+markers'
                    ), secondary_y=True)
                    
                    # Update layout with dual y-axis titles
                    fig.update_layout(
                        title='Displacement Trends',
                        hovermode='x unified',
                        showlegend=True,
                        barmode='stack'
                    )
                    
                    fig.update_yaxes(title_text="Individual Categories", secondary_y=False)
                    fig.update_yaxes(title_text="Total Displaced", secondary_y=True)
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No displacement data available for this country")
            else:
                # Check which specific columns are missing
                missing_cols = []
                for col_name, col in zip(
                    ["Refugee", "IDP", "Asylum Seeker", "Total Displaced"], 
                    [refugee_col, idp_col, asylum_col, total_col]
                ):
                    if col not in country_data.columns:
                        missing_cols.append(col_name)
                
                if missing_cols:
                    st.info(f"Missing data columns: {', '.join(missing_cols)}")
                else:
                    st.info("Displacement data columns exist but contain no data for this country")
        except Exception as e:
            st.warning(f"Could not display displacement analysis: {str(e)}")
            # Print detailed traceback for debugging
            import traceback
            st.error(traceback.format_exc())

    # Add fourth row of visualizations
    col9, col10 = st.columns(2)

    with col9:
        # Recent Conflict Trends (last 3 years)
        st.subheader("Recent Conflict Indicators")
        try:
            recent_data = pd.DataFrame({
                'Year': country_data['year'],
                'Conflicts Past 3 Years': country_data['conflicts_past_3years'],
                'Intensity Past 3 Years': country_data['intensity_past_3years']
            }).dropna()
            
            if not recent_data.empty:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=recent_data['Year'],
                    y=recent_data['Conflicts Past 3 Years'],
                    name='Conflicts (3-yr)',
                    line=dict(color='darkorange', width=2),
                    mode='lines+markers'
                ))
                
                fig.add_trace(go.Scatter(
                    x=recent_data['Year'],
                    y=recent_data['Intensity Past 3 Years'],
                    name='Intensity (3-yr)',
                    line=dict(color='darkred', width=2, dash='dot'),
                    mode='lines+markers',
                    yaxis='y2'
                ))
                
                fig.update_layout(
                    title='Recent Conflict Trends (3-Year Window)',
                    yaxis=dict(title='Number of Conflicts'),
                    yaxis2=dict(title='Intensity Scale', overlaying='y', side='right'),
                    hovermode='x unified',
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No recent conflict trend data available")
        except Exception as e:
            st.warning(f"Could not display recent conflict indicators: {str(e)}")

    with col10:
        # Military Spending Trends and Changes
        st.subheader("Military Spending Trends")
        try:
            military_data = pd.DataFrame({
                'Year': country_data['year'],
                'Military Spending (% GDP)': country_data['share_of_gdp'],
                'Military Spending (% Govt)': country_data['share_of_govt._spending'],
                '5yr Avg (% GDP)': country_data['share_of_gdp_5yr_avg']
            }).dropna()
            
            if not military_data.empty:
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=military_data['Year'],
                    y=military_data['Military Spending (% GDP)'],
                    name='% of GDP',
                    marker_color='navy'
                ))
                
                fig.add_trace(go.Scatter(
                    x=military_data['Year'],
                    y=military_data['Military Spending (% Govt)'],
                    name='% of Govt Spending',
                    line=dict(color='darkslategray'),
                    yaxis='y2'
                ))
                
                fig.add_trace(go.Scatter(
                    x=military_data['Year'],
                    y=military_data['5yr Avg (% GDP)'],
                    name='5yr Avg (% GDP)',
                    line=dict(color='royalblue', dash='dash'),
                    mode='lines'
                ))
                
                fig.update_layout(
                    title='Military Spending Analysis',
                    yaxis=dict(title='% of GDP'),
                    yaxis2=dict(title='% of Govt Spending', overlaying='y', side='right'),
                    hovermode='x unified',
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No military spending trend data available")
        except Exception as e:
            st.warning(f"Could not display military spending trends: {str(e)}")


    


    # Full width Conflict and Displacement Heatmap
    st.subheader("Conflict and Humanitarian Impact Heatmap")
    try:
        # Select conflict and humanitarian-related columns
        heatmap_columns = [
            'num_conflicts', 
            'avg_intensity', 
            'max_intensity',
            'conflicts_past_3years',
            'Political_Stability',
            'Total_Displaced',
            'Refugees under UNHCR\'s mandate',
            'IDPs of concern to UNHCR',
            'Unemployment, total (% of total labor force) (national estimate)'
        ]
        
        # Create a DataFrame for the heatmap with available columns
        available_columns = [col for col in heatmap_columns if col in country_data.columns]
        
        if len(available_columns) >= 4:  # Need at least a few columns for meaningful heatmap
            heatmap_data = country_data[['year'] + available_columns].copy()
            
            # Handle missing data - forward fill if possible
            heatmap_data = heatmap_data.sort_values('year').ffill().bfill()
            
            if not heatmap_data.empty:

                # Get the full range of years but select at intervals
                all_years = sorted(heatmap_data['year'].astype(int).unique())
                earliest_year = min(all_years)
                latest_year = max(all_years)
                
                # Calculate interval to show approximately 10-12 years on the x-axis
                total_years = latest_year - earliest_year + 1
                interval = max(1, total_years // 10)  # Ensure interval is at least 1
                
                # Select years at the calculated interval, always including earliest and latest
                selected_years = [all_years[i] for i in range(0, len(all_years), interval)]
                if latest_year not in selected_years:
                    selected_years.append(latest_year)
                
                # Filter data to only include the selected years
                display_data = heatmap_data[heatmap_data['year'].isin(selected_years)]
                display_data = display_data.sort_values('year')
                
                # Normalize data for better visualization (all columns on similar scale)
                normalized_data = heatmap_data.copy()
                for col in available_columns:
                    min_val = normalized_data[col].min()
                    max_val = normalized_data[col].max()
                    if max_val > min_val:  # Avoid division by zero
                        normalized_data[col] = (normalized_data[col] - min_val) / (max_val - min_val)
                
                # Create a pivot table for the heatmap
                z_data = []
                x_labels = heatmap_data['year'].astype(str).unique()
                
                # Generate heatmap data
                for col in available_columns:
                    row_data = []
                    for year in heatmap_data['year'].unique():
                        value = normalized_data.loc[normalized_data['year'] == year, col].values[0]
                        row_data.append(value)
                    z_data.append(row_data)
                
                # Create more readable y-axis labels
                y_labels = []
                for col in available_columns:
                    if col == 'num_conflicts':
                        y_labels.append('Number of Conflicts')
                    elif col == 'avg_intensity':
                        y_labels.append('Average Conflict Intensity')
                    elif col == 'max_intensity':
                        y_labels.append('Maximum Conflict Intensity')
                    elif col == 'conflicts_past_3years':
                        y_labels.append('Conflicts in Past 3 Years')
                    elif col == 'Political_Stability':
                        y_labels.append('Political Stability')
                    elif col == 'Total_Displaced':
                        y_labels.append('Total Displaced People')
                    elif col == 'Refugees under UNHCR\'s mandate':
                        y_labels.append('Refugees (UNHCR)')
                    elif col == 'IDPs of concern to UNHCR':
                        y_labels.append('Internally Displaced')
                    elif 'Unemployment' in col:
                        y_labels.append('Unemployment Rate')
                    else:
                        y_labels.append(col.replace('_', ' '))
                
                # Create custom hover text with actual values
                hover_text = []
                for i, col in enumerate(available_columns):
                    row_text = []
                    for year in heatmap_data['year'].unique():
                        original_value = heatmap_data.loc[heatmap_data['year'] == year, col].values[0]
                        if isinstance(original_value, (int, float)):
                            if original_value > 1000:
                                formatted_value = f"{original_value:,.0f}"
                            else:
                                formatted_value = f"{original_value:.2f}"
                        else:
                            formatted_value = str(original_value)
                        
                        row_text.append(f"{y_labels[i]}<br>Year: {year}<br>Value: {formatted_value}")
                    hover_text.append(row_text)
                
                # Create heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=z_data,
                    x=x_labels,
                    y=y_labels,
                    colorscale='RdBu_r',  # Red-Blue reversed (Red=High, Blue=Low)
                    colorbar=dict(title='Normalized Score'),
                    hoverinfo='text',
                    text=hover_text
                ))
                
                # Improved layout for full-width display
                fig.update_layout(
                    title='Conflict and Humanitarian Impact Over Time',
                    xaxis=dict(title='Year', type='category'),
                    yaxis=dict(title='Indicator', tickangle=0),
                    height=600,  # Increased height for better visibility
                    margin=dict(l=220, r=50, t=70, b=50)  # Increased left margin for labels
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add legend explaining the heatmap
                st.markdown("""
                <small>
                <b>How to read this heatmap:</b> Red indicates higher values (potentially more concerning), 
                blue indicates lower values. All metrics are normalized for comparison. 
                Hover over cells to see actual values.
                </small>
                """, unsafe_allow_html=True)
                
            else:
                st.info("Insufficient data for conflict and humanitarian heatmap")
        else:
            st.info("Not enough conflict and humanitarian data columns available")
    except Exception as e:
        st.warning(f"Could not display conflict and humanitarian heatmap: {str(e)}")


    # Add comprehensive conflict risk assessment card
    st.subheader("📊 Conflict Risk Assessment")
    try:
        
        
        # Get data for 2023, but prepare for interpolation
        all_years_data = country_data.sort_values('year')
        
        # Apply interpolation to the dataset to fill missing values
        interpolated_data = all_years_data.copy()
        
        # First try linear interpolation for numeric columns
        numeric_columns = interpolated_data.select_dtypes(include=['number']).columns
        interpolated_data[numeric_columns] = interpolated_data[numeric_columns].interpolate(method='linear', limit_direction='both')
        
        # Then use forward and backward fill for any remaining NaNs
        interpolated_data = interpolated_data.ffill().bfill()
        
        # Get the latest data (2023) from the interpolated dataset
        latest_data = interpolated_data[interpolated_data['year'] == 2023]
        
        if not latest_data.empty:
            latest_data = latest_data.iloc[0]
            
            # Create 3 columns for risk factors
            risk_col1, risk_col2, risk_col3 = st.columns(3)
            
            with risk_col1:
                # Historical conflict indicators
                num_conflicts = latest_data.get('num_conflicts')
                max_intensity = latest_data.get('max_intensity')
                
                if pd.notna(num_conflicts):
                    st.metric(
                        "Active Conflicts",
                        f"{int(num_conflicts)}",
                        f"Max Intensity: {max_intensity:.1f}" if pd.notna(max_intensity) else None
                    )
                    st.caption("(Interpolated data)" if pd.isna(all_years_data[all_years_data['year'] == 2023]['num_conflicts'].iloc[0]) else "")
                else:
                    st.metric("Active Conflicts", "No data", None)
                
                # Recent trend
                conflicts_3yr = latest_data.get('conflicts_past_3years')
                if pd.notna(conflicts_3yr):
                    trend = ""
                    if pd.notna(num_conflicts):
                        trend = "↑" if num_conflicts > conflicts_3yr/3 else "↓"
                    st.metric("3-Year Conflict Trend", f"{int(conflicts_3yr)} total", trend)
                    st.caption("(Interpolated data)" if pd.isna(all_years_data[all_years_data['year'] == 2023]['conflicts_past_3years'].iloc[0]) else "")
                else:
                    st.metric("3-Year Trend", "No data", None)
            
            with risk_col2:
                # Political stability indicators
                pol_stability = latest_data.get('Political_Stability')
                democracy = latest_data.get('Democracy_Score')
                
                if pd.notna(pol_stability):
                    color = "green"
                    if pol_stability < 0:
                        color = "red"
                    elif pol_stability < 3:
                        color = "orange"
                    
                    st.markdown(f"#### Political Stability")
                    st.markdown(f"<span style='color:{color};font-size:24px;font-weight:bold'>{pol_stability:.2f}</span>", unsafe_allow_html=True)
                    if pd.isna(all_years_data[all_years_data['year'] == 2023]['Political_Stability'].iloc[0]):
                        st.caption("(Interpolated data)")
                else:
                    st.metric("Political Stability", "No data", None)
                
                # Regime durability
                durability = latest_data.get('Regime_Durability_Years')
                if pd.notna(durability):
                    risk_level = "Low Risk"
                    color = "green"
                    if durability < 5:
                        risk_level = "High Risk"
                        color = "red"
                    elif durability < 10:
                        risk_level = "Medium Risk"
                        color = "orange"
                    
                    st.markdown(f"#### Regime Durability: <span style='color:{color}'>{risk_level}</span>", unsafe_allow_html=True)
                    st.markdown(f"<span style='font-size:18px'>{int(durability)} years</span>", unsafe_allow_html=True)
                    if pd.isna(all_years_data[all_years_data['year'] == 2023]['Regime_Durability_Years'].iloc[0]):
                        st.caption("(Interpolated data)")
                else:
                    st.metric("Regime Durability", "No data", None)
            
            with risk_col3:
                # Military indicators
                military_gdp = latest_data.get('share_of_gdp')
                mil_5yr_avg = latest_data.get('share_of_gdp_5yr_avg')
                
                if  pd.notna(military_gdp):
                    direction = "↑"
                    if pd.notna(mil_5yr_avg) and military_gdp < mil_5yr_avg:
                        direction = "↓"
                    
                    st.metric(
                        "Military Spending (% GDP)",
                        f"{military_gdp:.2f}%",
                        f"{direction} vs 5yr avg" if pd.notna(mil_5yr_avg) else None
                    )
                    if pd.isna(all_years_data[all_years_data['year'] == 2023]['share_of_gdp'].iloc[0]):
                        st.caption("(Interpolated data)")
                else:
                    st.metric("Military Spending", "No data", None)
                
                # Arms transfers
                arms_transfers = latest_data.get('Arms_Transfer_Value')
                if pd.notna(arms_transfers):
                    st.metric("Arms Transfers", f"${arms_transfers:,.2f}M", None)
                    if pd.isna(all_years_data[all_years_data['year'] == 2023]['Arms_Transfer_Value'].iloc[0]):
                        st.caption("(Interpolated data)")
                else:
                    st.metric("Arms Transfers", "No data", None)
            

            # We can now proceed with calculating risk as we've filled missing values
            # Political stability (reverse scale as lower stability means higher risk)
            composite_risk = 0
            factors = 0
            
            if pd.notna(pol_stability):
                if pol_stability < 0:
                    composite_risk += 10  # High risk
                elif pol_stability < 3:
                    composite_risk += 5   # Medium risk
                else:
                    composite_risk += 3   # Low risk
                factors += 1
            
            # Military spending
            if pd.notna(military_gdp):
                if military_gdp > 5:
                    composite_risk += 10  # High risk
                elif military_gdp > 3:
                    composite_risk += 5   # Medium risk
                else:
                    composite_risk += 2   # Low risk
                factors += 1
            
            # Active conflicts
            if pd.notna(num_conflicts):
                if num_conflicts >= 2:
                    composite_risk += 10  # High risk
                elif num_conflicts > 0:
                    composite_risk += 5   # Medium risk
                else:
                    composite_risk += 1   # Low risk
                factors += 1
            
            # Additional factors if available
            if pd.notna(durability):
                if durability < 5:
                    composite_risk += 10  # High risk
                elif durability < 10:
                    composite_risk += 5   # Medium risk
                else:
                    composite_risk += 2   # Low risk
                factors += 1
            
            # Calculate average risk (1-10 scale)
            if factors > 0:
                composite_risk = composite_risk / factors
                risk_level = "Low"
                color = "green"
                if composite_risk > 7:
                    risk_level = "High"
                    color = "red"
                elif composite_risk > 4:
                    risk_level = "Medium"
                    color = "orange"
                
                st.markdown(f"## Overall Conflict Risk Assessment: <span style='color:{color}'>{risk_level}</span>", unsafe_allow_html=True)
                
                # Show a note that interpolated data was used if necessary
                if any(pd.isna(all_years_data[all_years_data['year'] == 2023][['num_conflicts', 'Political_Stability', 'share_of_gdp', 'Regime_Durability_Years']].iloc[0])):
                    st.caption("Note: Some metrics use interpolated data for more complete assessment")
                
                # Create a gauge chart for overall risk
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = composite_risk,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Conflict Risk Score"},
                    gauge = {
                        'axis': {'range': [0, 10]},
                        'bar': {'color': color},
                        'steps': [
                            {'range': [0, 4], 'color': "lightgreen"},
                            {'range': [4, 7], 'color': "lightyellow"},
                            {'range': [7, 10], 'color': "salmon"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': composite_risk
                        }
                    }
                ))
                
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display information about which factors contributed to the risk score
                st.expander("Risk Factor Details").markdown(f"""
                **Factors considered in risk assessment:**
                - Political Stability: {'Low' if pol_stability >= 3 else 'Medium' if pol_stability >= 0 else 'High'} risk factor
                - Military Spending: {'Low' if military_gdp <= 3 else 'Medium' if military_gdp <= 5 else 'High'} risk factor
                - Active Conflicts: {'Low' if num_conflicts == 0 else 'Medium' if num_conflicts <= 2 else 'High'} risk factor
                - Regime Durability: {'Low' if durability >= 10 else 'Medium' if durability >= 5 else 'High'} risk factor
                
                **Composite Score:** {composite_risk:.2f}/10
                """)

            else:
                st.warning("Insufficient data for comprehensive conflict risk assessment")
    except Exception as e:
        st.warning(f"Could not display conflict risk assessment: {str(e)}")

    # Add summary statistics
    st.subheader("📈 Key Statistics")
    try:
        # latest_year = country_data[[year]==2023]
        latest_data = country_data[country_data['year'] == 2023]
        
        if not latest_data.empty:
            latest_data = latest_data.iloc[0]
            # Create three columns for statistics
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            
            with stat_col1:
                gdp_per_capita = latest_data.get('GDP per capita (current US$)', 'N/A')
                gdp_growth = latest_data.get('constant_(2022)_us$', 'N/A')

                if gdp_per_capita != 'N/A':
                    st.metric(
                        "Latest GDP per capita",
                        f"${gdp_per_capita:,.2f}",
                        f"{gdp_growth:.1f}%" if gdp_growth != 'N/A' else None
                    )
                else:
                    st.metric("Latest GDP per capita", "No data", None)
            

            with stat_col2:
                military_share_of_gdp = latest_data.get('share_of_gdp', 'N/A')
                arms_trade = latest_data.get('Arms_Transfer_Value', 'N/A')

                if military_share_of_gdp != 'N/A':
                    st.metric(
                        "Military Expenditure",
                        f"{military_share_of_gdp:.1f}%",
                        f"${arms_trade:,.2f}" if arms_trade != 'N/A' else None
                    )
                else:
                        st.metric("Military data ", "No data", None)
            
            with stat_col3:
                unemployment = latest_data.get('Unemployment, total (% of total labor force) (national estimate)', 'N/A')
                education_expense = latest_data.get('Adjusted savings: education expenditure (current US$)', 'N/A')
                if unemployment != 'N/A':
                    st.metric(
                        "Political Stability Score",
                        f"{unemployment:.2f}",
                        f"{education_expense:.2f}" if education_expense != 'N/A' else None
                    )
                else:
                    st.metric("Unemployment", "No data", None)
    except Exception as e:
        st.warning(f"Could not display summary statistics: {str(e)}")



def main():
    st.title("🌍 Global Conflict Hotspots Prediction")
    
    
    # Directly load CSV from local path
    csv_file_path = r"D:\datavisual-claude\cleaned\merged_conflict_dataset.csv"  # Change this path as needed
    df = load_and_prepare_data(csv_file_path)
    
    # if uploaded_file is not None:
    try:
            # df = load_and_prepare_data(uploaded_file)
        
            if df is not None:
                
                

                #new code
                predictions = train_predictive_model(df)
                df_final = aggregate_predictions(predictions)
                
                

                #new code
                deck = create_globe_visualization(df_final)
                st.pydeck_chart(deck)
                
                # with col2:
                st.subheader("Country Analysis")
                # Get unique countries and sort them
                countries = sorted(df['country'].unique())
                    
                # Create country selector
                selected_country = st.selectbox(
                    "Select a country for detailed analysis:",
                    countries,
                    index=None,
                    placeholder="Choose a country..."
                )
                    
                #old code
                if selected_country:
                    # Show "View Details" button
                    # if st.button(f"View Details for {selected_country}"):
                        st.session_state.selected_country = selected_country
                        st.session_state.show_analysis = True

                
                
                #old code
                # Show country analysis if selected
                if 'show_analysis' in st.session_state and st.session_state.show_analysis:
                    create_country_page(df, st.session_state.selected_country)
                    if st.button("← Back to Global View"):
                        st.session_state.show_analysis = False
                        st.rerun()
                        
    except Exception as e:
                st.error(f"Error processing data: {str(e)}")
            

if __name__ == "__main__":
    main()