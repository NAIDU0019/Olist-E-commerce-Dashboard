

# 📊 Olist E-commerce Dashboard

An interactive data science project analyzing the Olist Brazilian E-commerce dataset. This dashboard uncovers business insights across customer behavior, sales performance, delivery efficiency, and product popularity. It also includes machine learning for price prediction based on product attributes.

VISIT LIVE:https://oliste-commercedashboard.streamlit.app/

---

## 📁 Dataset

The project is based on the [Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), containing detailed data about:

* Customers
* Orders
* Payments
* Products
* Reviews
* Sellers
* Geolocation

Data spans from 2016 to 2018 and includes over 100,000 orders across various product categories.

---

## 🧠 Features & Workflow

### 🧹 Data Cleaning & Preprocessing

* Merged multiple datasets into a unified structure.
* Removed duplicates and handled missing values.
* Translated product categories to English.
* Engineered features like:

  * `product_volume_cm3` (dimensional product volume)
  * `price_bucket` (categorical price ranges)
  * Category popularity based on order frequency

### 📊 Exploratory Data Analysis (EDA)

* Visualized top-selling product categories.
* Analyzed pricing distribution and review scores.
* Generated a comprehensive `ydata_profiling` EDA report.

### 🛠 Feature Engineering

* Key features used:

  * `product_name_length`
  * `product_description_length`
  * `product_photos_qty`
  * `product_weight_g`
  * `product_volume_cm3`
  * `category_popularity`
* Target variable: `price_bucket`

### 🤖 Machine Learning

* Model: Random Forest Classifier
* Task: Predict product price buckets
* Evaluated using accuracy, precision, recall, and F1-score

### 📈 Streamlit Dashboard

Interactive web dashboard with the following pages:

* **Overview**: Dataset summary and cleaning process
* **Category Insights**: Popular categories and their stats
* **Price Analysis**: Price bucket distribution and volume insights
* **Model Performance**: Classifier metrics and predictions

---

## 🧰 Tech Stack

* **Python**
* **Pandas, NumPy** – data manipulation
* **Matplotlib, Seaborn** – static visualization
* **ydata-profiling** – automated EDA report
* **Scikit-learn** – machine learning
* **Streamlit** – dashboard frontend
* **Jupyter Notebook** – development environment

---

## 🚀 Getting Started

Clone this repository:

```bash
git clone https://github.com/NAIDU0019/Olist-E-commerce-Dashboard.git
cd Olist-Ecommerce-Dashboard
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```

To explore the notebooks:

```bash
jupyter notebook
```

---

## 📷 Sample Visuals


---

## ✅ Results

* Improved understanding of pricing patterns by category and product type.
* Identified features that influence price ranges.
* Built a reliable classification model with interpretable results.
* Delivered a clean, interactive dashboard accessible to non-technical stakeholders.

---

## 🤝 Contributing

Pull requests and feature suggestions are welcome. Feel free to fork the project and improve it further!

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 📌 Related Work & Inspiration

* [Olist E-Commerce Analysis by Sameera Kota](https://github.com/SameeraKota/Olist-E-Commerce-Analysis)
* [Tableau Dashboard by Anna Barentz](https://public.tableau.com/app/profile/annabarentz/viz/E-CommerceDashboardOlist/Dashboard3)
* [Comprehensive Analysis by Raktim Mazumdar](https://github.com/raktimmazumdar/Data-Analysis-for-Olist-Brazilian-E-Commerce)
