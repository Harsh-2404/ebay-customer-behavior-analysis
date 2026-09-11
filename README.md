# 🛒 eBay Customer Purchasing Behavior & Recommendation Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458.svg)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ebay-customer-behavior-analysis-harsh.streamlit.app/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/Harsh-2404/ebay-customer-behavior-analysis/blob/main/eBay_Customer_Purchasing_Behavior_Analysis.ipynb)


---

## 📌 Project Overview
This project provides an end-to-end data analytics and machine learning workflow on customer purchasing behavior, recommendation engine efficacy, search methodologies, and shopping cart abandonment factors on the **eBay** e-commerce platform.

Using survey responses from **800 eBay customers**, this analysis uncovers key friction points in the conversion funnel, segments users into actionable behavioral personas using **K-Means Clustering**, and evaluates how customer review trust impacts platform satisfaction.

---

## 📊 Dataset Summary
* **Source File:** `eBay.csv`
* **Total Entries:** 800 rows
* **Total Features:** 24 columns
* **Target Domains:** Demographics, Browsing & Purchase Frequency, Search Preferences, Cart Abandonment Reasons, Review Reliability, and Recommendation Satisfaction.

### Core Attributes Covered:
| Attribute Category | Features |
| :--- | :--- |
| **Demographics** | `age`, `Gender` |
| **Shopping Habits** | `Purchase_Frequency`, `Browsing_Frequency`, `Purchase_Categories` |
| **Search & Discovery**| `Product_Search_Method`, `Search_Result_Exploration` |
| **Checkout Dynamics**| `Cart_Completion_Frequency`, `Cart_Abandonment_Factors` |
| **Feedback & Ratings**| `Customer_Reviews_Importance`, `Rating_Accuracy`, `Shopping_Satisfaction`, `Recommendation_Helpfulness` |

---

## 🛠️ Key Technical Deliverables & Methodology

### Task 1: Data Cleaning & Preprocessing
* **Duplicate Analysis:** Inspected 800 entries; verified **0 duplicate rows**.
* **Missing Value Imputation:** Handled `Product_Search_Method` missing values (~20.12% / 161 entries) by categorizing them under `"Unknown"`. Standardized missing placeholder symbols (e.g., `.`) in feedback columns.
* **Column Renaming & Standardization:** Standardized misformatted headers with trailing whitespaces (e.g., `Rating_Accuracy`). Disambiguated duplicate recommendation features by renaming the numerical rating feature to `Personalized_Recommendation_Rating`.
* **Type Casting:** Converted survey scale ratings (`Customer_Reviews_Importance`, `Shopping_Satisfaction`, etc.) to standard integer numeric types (`int64`).

### Task 2: Exploratory Data Analysis (EDA)
* **Age Distribution:** Customer age spans from **3 to 67 years** (Mean Age: **~35.73 years**, Median Age: **37.0 years**).
* **Gender Proportions:** Extremely balanced across Male (23.9%), Female (24.8%), Others (26.1%), and Prefer not to say (25.2%).
* **Primary Search Methods:** **Keyword Search** is the most preferred product discovery method (175 users), followed by **Category Browsing** (158 users) and **Filters** (142 users).
* **Top Cart Abandonment Drivers:**
  1. **High Shipping Costs** (208 customers)
  2. **Changed Mind / Item No Longer Needed** (204 customers)
  3. **Found Better Price Elsewhere** (184 customers)

### Task 3: Customer Segmentation (K-Means Machine Learning)
Using `LabelEncoder` and `StandardScaler` on behavioral variables (`age`, `Purchase_Frequency`, `Shopping_Satisfaction`, `Cart_Completion_Frequency`), the **Elbow Method** determined an optimal **$K = 3$** clusters:

* **Cluster 0: Older Low-Satisfaction Shoppers** (Mean Age: ~43.3 years | Satisfaction: 2.14/5.0)
  * Low rating satisfaction, moderate purchase frequency, higher tendency to abandon carts.
* **Cluster 1: Young High-Satisfaction Power Users** (Mean Age: ~28.8 years | Satisfaction: 4.36/5.0)
  * High cart completion rates, frequent purchases, highly responsive to personalized recommendations.
* **Cluster 2: Price-Sensitive Middle-Aged Buyers** (Mean Age: ~36.1 years | Satisfaction: 1.89/5.0)
  * Highly sensitive to shipping costs, occasional buyers with low rating accuracy expectations.

### Task 4: Recommendation Engine & Review Insights
* **Recommendation Satisfaction:** Users who perceive recommendations as helpful show a higher baseline of platform satisfaction (~2.90 mean satisfaction score).
* **Review Trust:** Strong positive relationship between `Customer_Reviews_Importance` and expected `Rating_Accuracy` (users valuing reviews highest demand higher item description precision, averaging 3.23 in rating accuracy).

---

## 📈 Key Visualizations Produced
1. **Demographic Histograms & Pie Charts:** Visualizing age spread and gender distribution across platform users.
2. **Ranked Bar Charts:** Identifying primary checkout friction points (Shipping Costs vs. Price Competition).
3. **Scatter & Box Plots:** Clustering scatter plots illustrating distinct customer personas across satisfaction levels and age groups.
4. **Correlation Heatmaps:** Showing feature relationships between review importance, recommendation helpfulness, and overall satisfaction.

---

## 💡 Strategic Business Recommendations

1. **Mitigate Cart Abandonment:**
   * Implement **dynamic free-shipping thresholds** to address the #1 abandonment factor (High Shipping Costs).
   * Send automated exit-intent popups or abandoned cart discount triggers for users comparing prices.
2. **Optimize ML Personalization Engines:**
   * Tailor recommendation models specifically for Cluster 0 & Cluster 2 buyers using past keyword search histories rather than broad category trends.
   * Feature context-relevant recommendations directly on item pages to capture power users (Cluster 1).
3. **Elevate Review Transparency & Trust:**
   * Highlight **Verified Buyer** badges prominently on product reviews to satisfy high-review-importance shoppers.
   * Encourage structured post-purchase feedback to increase overall product rating precision.
4. **UI/UX Search Enhancements:**
   * Improve auto-complete suggestions for Keyword Searches and simplify category filter navigation.

---

## 📁 Repository Directory Structure

```text
├── README.md                                         # Comprehensive Project Documentation
├── eBay.csv                                         # Raw Survey Dataset (800 rows, 24 attributes)
├── cleaned.eBay.csv                                  # Cleaned Dataset
├── eBay_Customer_Purchasing_Behavior_Analysis.ipynb  # Complete Exploratory & Predictive Data Analysis
├── eBay_Customer_Purchasing_Behavior_Analysis.pdf    # Official PDF Report (Visual Insights & Executive Brief)
└── eBay Project Summary Report.docx                  # Official Business & Technical Summary
