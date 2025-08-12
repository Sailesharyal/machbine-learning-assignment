# import joblib
# import streamlit as st
# import numpy as np
# import pandas as pd

# # Load models
# models = {
#     "Decision Tree": joblib.load("decisiontree_model.pkl"),
#     "XGBoost": joblib.load("xgboost_model.pkl"),
#     "CatBoost": joblib.load("catboost_model.pkl"),
#     "LG Boost": joblib.load("lightgbm_model.pkl"),
#     "Aada Boost": joblib.load("adaboost_model.pkl"),
# }

# # Define ensemble weights
# model_weights = {
#     "XGBoost": 0.185013,
#     "LG Boost": 0.000181,
#     "CatBoost": 0.012551,
#     "Aada Boost": 0.153472,
#     "Decision Tree": 0.023876,
# }

# # Load encoders
# brand_encoder = joblib.load("brand_encoder.pkl")
# fuel_type_encoder = joblib.load("fuel_type_encoder.pkl")
# transmission_encoder = joblib.load("transmission_encoder.pkl")
# ext_col_encoder = joblib.load("ext_col_encoder.pkl")
# int_col_encoder = joblib.load("int_col_encoder.pkl")
# accident_encoder = joblib.load("accident_encoder.pkl")
# clean_title_encoder = joblib.load("clean_title_encoder.pkl")

# # Page configuration
# st.set_page_config(
#     page_title="Car Price Predictor", 
#     page_icon="🚗", 
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for enhanced styling
# st.markdown("""
# <style>
#     .big-font {
#         font-size:30px !important;
#         font-weight: bold;
#         color: #1f77b4;
#     }
#     .medium-font {
#         font-size:20px !important;
#         font-weight: bold;
#         color: #ff7f0e;
#     }
#     .metric-card {
#         background-color: #f0f2f6;
#         border: 2px solid #e0e0e0;
#         padding: 1rem;
#         border-radius: 10px;
#         text-align: center;
#     }
#     .stSelectbox > div > div {
#         background-color: #f8f9fa;
#     }
#     .prediction-container {
#         background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
#         padding: 2rem;
#         border-radius: 15px;
#         text-align: center;
#         color: white;
#         margin: 1rem 0;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Main header
# st.markdown('<p class="big-font">🚗 AI-Powered Car Price Prediction System</p>', unsafe_allow_html=True)
# st.markdown("---")

# # Description
# st.info("💡 **How it works:** Enter your car details below and let our advanced machine learning models estimate the market value of your vehicle!")

# # Sidebar for model selection
# with st.sidebar:
#     st.markdown("### 🎯 Model Selection Center")
#     st.markdown("Choose your preferred prediction model:")
    
#     model_choice = st.radio(
#         "Select Model:",
#         ["🌟 Ensemble Model (Recommended)"] + [f"🤖 {name}" for name in models.keys()],
#         index=0
#     )
    
#     # Clean model choice
#     if "Ensemble" in model_choice:
#         display_model = "Ensemble Model"
#         actual_model = "Ensemble Model"
#     else:
#         display_model = model_choice.replace("🤖 ", "")
#         actual_model = display_model
    
#     st.markdown("---")
#     st.markdown("### 📊 Model Information")
    
#     if "Ensemble" in model_choice:
#         st.success("✨ **Ensemble Model Selected**")
#         st.write("This model combines 9 different ML algorithms with optimized weights for maximum accuracy.")
        
#         with st.expander("View Model Weights"):
#             weight_df = pd.DataFrame.from_dict(model_weights, orient='index', columns=['Weight'])
#             weight_df['Weight'] = weight_df['Weight'].round(3)
#             st.dataframe(weight_df, use_container_width=True)
#     else:
#         st.info(f"🎯 **{display_model}** Selected")
#         st.write(f"Using {display_model} algorithm for prediction.")

# # Main input section
# st.markdown('<p class="medium-font">📝 Enter Car Details</p>', unsafe_allow_html=True)

# # Create tabs for better organization
# tab1, tab2, tab3 = st.tabs(["🚗 Basic Info", "🎨 Appearance & History", "⚙️ Technical Specs"])

# with tab1:
#     col1, col2 = st.columns(2)
    
#     with col1:
#         brand = st.selectbox("🏭 **Brand**", brand_encoder.classes_, help="Select the car manufacturer")
#         fuel_type = st.selectbox("⛽ **Fuel Type**", fuel_type_encoder.classes_, help="Choose fuel type")
#         transmission = st.selectbox("🔧 **Transmission**", transmission_encoder.classes_, help="Select transmission type")
    
#     with col2:
#         mileage = st.number_input(
#             "🛣️ **Total Kilometers Travelled**", 
#             min_value=0, 
#             value=50000, 
#             step=1000,
#             help="Enter total mileage in kilometers"
#         )
#         model_age = st.number_input(
#             "📅 **Car Age (years)**", 
#             min_value=0, 
#             max_value=50,
#             value=5, 
#             help="How old is your car?"
#         )

# with tab2:
#     col1, col2 = st.columns(2)
    
#     with col1:
#         ext_col = st.selectbox("🎨 **Exterior Color**", ext_col_encoder.classes_, help="Choose exterior color")
#         int_col = st.selectbox("🪑 **Interior Color**", int_col_encoder.classes_, help="Select interior color")
    
#     with col2:
#         accident = st.selectbox("🚨 **Accident History**", accident_encoder.classes_, help="Any accident history?")
#         clean_title = st.selectbox("📋 **Clean Title Status**", clean_title_encoder.classes_, help="Clean title status")

# with tab3:
#     col1, col2 = st.columns(2)
    
#     with col1:
#         horsepower = st.number_input(
#             "💪 **Horsepower (HP)**", 
#             min_value=0, 
#             value=200,
#             step=10,
#             help="Engine horsepower"
#         )
#         displacement = st.number_input(
#             "🔧 **Engine Displacement (L)**", 
#             min_value=0.0, 
#             value=2.0,
#             step=0.1,
#             format="%.1f",
#             help="Engine displacement in liters"
#         )
    
#     with col2:
#         cylinder_count = st.number_input(
#             "🔩 **Number of Cylinders**", 
#             min_value=1, 
#             max_value=12,
#             value=4,
#             step=1,
#             help="Number of engine cylinders"
#         )

# # Function to safely transform categorical features
# def safe_transform(encoder, value):
#     try:
#         return encoder.transform([value])[0]
#     except ValueError:
#         return -1  # Assign -1 for unseen labels

# # Prediction section
# st.markdown("---")
# st.markdown('<p class="medium-font">🔮 Price Prediction</p>', unsafe_allow_html=True)

# col1, col2, col3 = st.columns([1, 2, 1])

# with col2:
#     if st.button("🚀 **Predict Car Price**", type="primary", use_container_width=True):
#         with st.spinner("🤖 AI is analyzing your car... Please wait!"):
#             # Encode categorical features
#             encoded_features = [
#                 safe_transform(brand_encoder, brand), mileage,
#                 safe_transform(fuel_type_encoder, fuel_type),
#                 safe_transform(transmission_encoder, transmission),
#                 safe_transform(ext_col_encoder, ext_col),
#                 safe_transform(int_col_encoder, int_col),
#                 safe_transform(accident_encoder, accident),
#                 safe_transform(clean_title_encoder, clean_title),
#                 horsepower, displacement, cylinder_count, model_age
#             ]
            
#             # Convert features into a DataFrame
#             feature_columns = ['brand', 'milage', 'fuel_type', 'transmission', 'ext_col', 'int_col', 'accident', 'clean_title', 'Horsepower', 'Displacement', 'Cylinder Count', 'model_age']
#             features_df = pd.DataFrame([encoded_features], columns=feature_columns)
            
#             # Make prediction with selected model
#             if actual_model == "Ensemble Model":
#                 weighted_sum = 0
#                 weight_total = sum(model_weights.values())

#                 for name, model in models.items():
#                     if name in model_weights:
#                         weighted_sum += model_weights[name] * model.predict(features_df)[0]

#                 predicted_price = weighted_sum / weight_total
#             else:
#                 predicted_price = models[actual_model].predict(features_df)[0]
        
#         # Display results
#         st.markdown("### 📊 Prediction Results")
        
#         if predicted_price < 0:
#             st.error(f"💰 **Predicted Car Price: ${predicted_price:,.2f}**")
#             st.warning("⚠️ **Warning:** The predicted price seems unrealistic. Please double-check your input values.")
#         else:
#             # Main prediction display
#             st.markdown(f"""
#                 <div class="prediction-container">
#                     <h2>💰 Estimated Car Price</h2>
#                     <h1>${predicted_price:,.2f}</h1>
#                     <p>Predicted using {display_model}</p>
#                 </div>
#             """, unsafe_allow_html=True)
            
#             # Additional insights
#             st.markdown("### 📈 Additional Insights")
            
#             col1, col2, col3, col4 = st.columns(4)
            
#             with col1:
#                 lower_bound = predicted_price * 0.9
#                 upper_bound = predicted_price * 1.1
#                 st.metric(
#                     "Price Range",
#                     f"${predicted_price:,.0f}",
#                     f"±${predicted_price * 0.1:,.0f}"
#                 )
            
#             with col2:
#                 price_per_km = predicted_price / max(mileage, 1) * 1000
#                 st.metric(
#                     "Price per 1000km", 
#                     f"${price_per_km:.2f}",
#                     "Value retention"
#                 )
            
#             with col3:
#                 price_per_hp = predicted_price / max(horsepower, 1)
#                 st.metric(
#                     "Price per HP",
#                     f"${price_per_hp:.0f}",
#                     "Performance value"
#                 )
            
#             with col4:
#                 yearly_depreciation = (predicted_price * 0.15 * model_age) if model_age > 0 else 0
#                 st.metric(
#                     "Age Impact",
#                     f"-${yearly_depreciation:,.0f}",
#                     f"{model_age} years old"
#                 )
            
#             # Summary information
#             st.info(f"""
#             📋 **Summary:** Your {brand} {fuel_type} with {mileage:,} km and {horsepower} HP is estimated at **${predicted_price:,.2f}**. 
#             This prediction is based on the car's specifications, condition, and market trends.
#             """)

# # Footer
# st.markdown("---")
# st.markdown("""
#     <div style='text-align: center; color: #666666; padding: 20px;'>
#         <p>🤖 <b>Powered by Advanced Machine Learning</b> | Built with ❤️ using Streamlit</p>
#         <p><i>Disclaimer: Predictions are estimates based on historical data and should be used as a reference guide only.</i></p>
#     </div>
# """, unsafe_allow_html=True)





import joblib
import streamlit as st
import numpy as np
import pandas as pd

# Load models
models = {
    "Decision Tree": joblib.load("decisiontree_model.pkl"),
    "XGBoost": joblib.load("xgboost_model.pkl"),
    "CatBoost": joblib.load("catboost_model.pkl"),
    "LG Boost": joblib.load("lightgbm_model.pkl"),
    "Aada Boost": joblib.load("adaboost_model.pkl"),
}

# Load encoders
brand_encoder = joblib.load("brand_encoder.pkl")
fuel_type_encoder = joblib.load("fuel_type_encoder.pkl")
transmission_encoder = joblib.load("transmission_encoder.pkl")
ext_col_encoder = joblib.load("ext_col_encoder.pkl")
int_col_encoder = joblib.load("int_col_encoder.pkl")
accident_encoder = joblib.load("accident_encoder.pkl")
clean_title_encoder = joblib.load("clean_title_encoder.pkl")

# Page configuration
st.set_page_config(
    page_title="Car Price Predictor", 
    page_icon="🚗", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
        color: #1f77b4;
    }
    .medium-font {
        font-size:20px !important;
        font-weight: bold;
        color: #ff7f0e;
    }
    .metric-card {
        background-color: #f0f2f6;
        border: 2px solid #e0e0e0;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
    .prediction-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<p class="big-font">🚗 AI-Powered Car Price Prediction System</p>', unsafe_allow_html=True)
st.markdown("---")

# Description
st.info("💡 **How it works:** Enter your car details below and let our advanced machine learning models estimate the market value of your vehicle!")

# Sidebar for model selection
with st.sidebar:
    st.markdown("### 🎯 Model Selection Center")
    st.markdown("Choose your preferred prediction model:")
    
    model_choice = st.radio(
        "Select Model:",
        [f"🤖 {name}" for name in models.keys()],
        index=0
    )
    
    # Clean model choice to remove emoji for dictionary key
    actual_model = model_choice.replace("🤖 ", "")
    
    st.markdown("---")
    st.markdown("### 📊 Model Information")
    st.info(f"🎯 **{actual_model}** Selected")
    st.write(f"Using {actual_model} algorithm for prediction.")

# Main input section
st.markdown('<p class="medium-font">📝 Enter Car Details</p>', unsafe_allow_html=True)

# Create tabs for better organization
tab1, tab2, tab3 = st.tabs(["🚗 Basic Info", "🎨 Appearance & History", "⚙️ Technical Specs"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        brand = st.selectbox("🏭 **Brand**", brand_encoder.classes_, help="Select the car manufacturer")
        fuel_type = st.selectbox("⛽ **Fuel Type**", fuel_type_encoder.classes_, help="Choose fuel type")
        transmission = st.selectbox("🔧 **Transmission**", transmission_encoder.classes_, help="Select transmission type")
    
    with col2:
        mileage = st.number_input(
            "🛣️ **Total Kilometers Travelled**", 
            min_value=0, 
            value=50000, 
            step=1000,
            help="Enter total mileage in kilometers"
        )
        model_age = st.number_input(
            "📅 **Car Age (years)**", 
            min_value=0, 
            max_value=50,
            value=5, 
            help="How old is your car?"
        )

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        ext_col = st.selectbox("🎨 **Exterior Color**", ext_col_encoder.classes_, help="Choose exterior color")
        int_col = st.selectbox("🪑 **Interior Color**", int_col_encoder.classes_, help="Select interior color")
    
    with col2:
        accident = st.selectbox("🚨 **Accident History**", accident_encoder.classes_, help="Any accident history?")
        clean_title = st.selectbox("📋 **Clean Title Status**", clean_title_encoder.classes_, help="Clean title status")

with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        horsepower = st.number_input(
            "💪 **Horsepower (HP)**", 
            min_value=0, 
            value=200,
            step=10,
            help="Engine horsepower"
        )
        displacement = st.number_input(
            "🔧 **Engine Displacement (L)**", 
            min_value=0.0, 
            value=2.0,
            step=0.1,
            format="%.1f",
            help="Engine displacement in liters"
        )
    
    with col2:
        cylinder_count = st.number_input(
            "🔩 **Number of Cylinders**", 
            min_value=1, 
            max_value=12,
            value=4,
            step=1,
            help="Number of engine cylinders"
        )

# Function to safely transform categorical features
def safe_transform(encoder, value):
    try:
        return encoder.transform([value])[0]
    except ValueError:
        return -1  # Assign -1 for unseen labels

# Prediction section
st.markdown("---")
st.markdown('<p class="medium-font">🔮 Price Prediction</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("🚀 **Predict Car Price**", type="primary", use_container_width=True):
        with st.spinner("🤖 AI is analyzing your car... Please wait!"):
            # Encode categorical features
            encoded_features = [
                safe_transform(brand_encoder, brand), mileage,
                safe_transform(fuel_type_encoder, fuel_type),
                safe_transform(transmission_encoder, transmission),
                safe_transform(ext_col_encoder, ext_col),
                safe_transform(int_col_encoder, int_col),
                safe_transform(accident_encoder, accident),
                safe_transform(clean_title_encoder, clean_title),
                horsepower, displacement, cylinder_count, model_age
            ]
            
            # Convert features into a DataFrame
            feature_columns = ['brand', 'milage', 'fuel_type', 'transmission', 'ext_col', 'int_col', 'accident', 'clean_title', 'Horsepower', 'Displacement', 'Cylinder Count', 'model_age']
            features_df = pd.DataFrame([encoded_features], columns=feature_columns)
            
            # Make prediction with selected model
            predicted_price = models[actual_model].predict(features_df)[0]
        
        # Display results
        st.markdown("### 📊 Prediction Results")
        
        if predicted_price < 0:
            st.error(f"💰 **Predicted Car Price: ${predicted_price:,.2f}**")
            st.warning("⚠️ **Warning:** The predicted price seems unrealistic. Please double-check your input values.")
        else:
            # Main prediction display
            st.markdown(f"""
                <div class="prediction-container">
                    <h2>💰 Estimated Car Price</h2>
                    <h1>${predicted_price:,.2f}</h1>
                    <p>Predicted using {actual_model}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Additional insights
            st.markdown("### 📈 Additional Insights")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Price Range",
                    f"${predicted_price:,.0f}",
                    f"±${predicted_price * 0.1:,.0f}"
                )
            
            with col2:
                price_per_km = predicted_price / max(mileage, 1) * 1000
                st.metric(
                    "Price per 1000km", 
                    f"${price_per_km:.2f}",
                    "Value retention"
                )
            
            with col3:
                price_per_hp = predicted_price / max(horsepower, 1)
                st.metric(
                    "Price per HP",
                    f"${price_per_hp:.0f}",
                    "Performance value"
                )
            
            with col4:
                yearly_depreciation = (predicted_price * 0.15 * model_age) if model_age > 0 else 0
                st.metric(
                    "Age Impact",
                    f"-${yearly_depreciation:,.0f}",
                    f"{model_age} years old"
                )
            
            # Summary information
            st.info(f"""
            📋 **Summary:** Your {brand} {fuel_type} with {mileage:,} km and {horsepower} HP is estimated at **${predicted_price:,.2f}**. 
            This prediction is based on the car's specifications, condition, and market trends.
            """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666666; padding: 20px;'>
        <p>🤖 <b>Powered by Advanced Machine Learning</b> | Built with ❤️ using Streamlit</p>
        <p><i>Disclaimer: Predictions are estimates based on historical data and should be used as a reference guide only.</i></p>
    </div>
""", unsafe_allow_html=True)
