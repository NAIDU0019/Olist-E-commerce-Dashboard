import streamlit as st
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from model_utils import train_and_evaluate_model
    # Load the cleaned data
CLEANED_PATH = 'cleaned_data/cleaned_products.csv'
merged_df = pd.read_csv(CLEANED_PATH)
top_cats = merged_df['product_category_name_english'].value_counts().head(10)
avg_price_per_category = merged_df.groupby('product_category_name_english')['price'].mean().sort_values(ascending=False)
top_categories = merged_df['product_category_name_english'].value_counts().head(10)
price_bucket_variation = merged_df.groupby('price_bucket')['price'].agg(['mean', 'std'])





# === Prepare Data ===
# Select features and target


# === Streamlit App ===
st.title("Olist E-commerce Dashboard")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Select a Page", ["Overview", "Category Insights", "Price Analysis", "Model Performance"])

# Overview Page
if page == "Overview":
    st.header("Overview")
    st.write("### Cleaned Data Summary")
    st.write(merged_df.describe())
    st.write("### Top 10 Categories by Product Count")
    st.bar_chart(top_cats)

# Category Insights Page
elif page == "Category Insights":
    st.header("Category Insights")
    st.write("### Average Price per Category")
    st.bar_chart(avg_price_per_category)
    st.write("### Top 10 Most Popular Categories")
    st.bar_chart(top_categories)

# Price Analysis Page
elif page == "Price Analysis":
    st.header("Price Analysis")
    st.write("### Price Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(merged_df['price'], bins=50, kde=True, color='skyblue', ax=ax)
    ax.set_title("Price Distribution After Cleaning")
    ax.set_xlabel("Price (R$)")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.write("### Price Variation Across Buckets")
    st.write(price_bucket_variation)

# Model Performance Page
elif page == "Model Performance":
    st.header("Model Performance")
    features = ['product_name_lenght', 'product_description_lenght', 'product_photos_qty', 
                'product_weight_g', 'product_volume_cm3', 'category_popularity']
    target = 'price_bucket'

    # Train and evaluate the model
    results = train_and_evaluate_model(merged_df, features, target)

    # Display classification report
    st.write("### Classification Report")
    st.text(results["classification_report"])

    # Display accuracy
    st.write("### Accuracy")
    st.metric("Accuracy", f"{results['accuracy']:.2%}")

    # Display feature importance
    st.write("### Feature Importance")
    feature_importance = pd.DataFrame({
        "Feature": features,
        "Importance": results["model"].feature_importances_
    }).sort_values(by="Importance", ascending=False)
    st.bar_chart(feature_importance.set_index("Feature"))

    # SHAP Values
    

    


# Run the app with `streamlit run app.py`
