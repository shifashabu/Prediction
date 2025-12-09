# Amazon Sales Prediction: Linear Regression

## 📌 Project Overview
This project demonstrates the **end-to-end process of predicting TotalAmount from Amazon sales data** using Linear Regression.  
The goal was to leverage cleaned sales data to build a predictive model, interpret feature importance, and evaluate model accuracy for informed business decisions.

---

## 🛠️ Tech Stack
- **Languages:** Python  
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn  
- **Tools:** VS Code, Jupyter Notebook  

---

## 🔍 Data Analysis & Modeling Steps
1. **Load Cleaned Dataset:**  
   - Read `Amazon_clean.csv` into a Pandas DataFrame.  
2. **Exploratory Data Analysis (EDA):**  
   - Quick data inspection with `head()` and `info()`.  
   - Visualized correlations using a heatmap to understand feature relationships.  
3. **Feature Selection:**  
   - Selected numeric features: `Quantity`, `UnitPrice`, `Discount`.  
   - Target variable: `TotalAmount`.  
4. **Feature Importance:**  
   - Calculated coefficients from a linear regression model to rank feature influence.  
   - Plotted feature importance for visual interpretation.  
5. **Train/Test Split:**  
   - Split dataset into training (80%) and testing (20%) sets.  
6. **Model Building:**  
   - Trained a Linear Regression model on the training data.  
7. **Evaluation:**  
   - Calculated R² Score and RMSE to measure model performance.  
   - Visualized predicted vs actual TotalAmount with a scatter plot.

---

## 📊 Visualizations
- **Correlation Heatmap:** Shows the relationships between features and the target.  
- **Feature Importance:** Highlights which features most influence TotalAmount.  
- **Predicted vs Actual Plot:** Assesses model prediction accuracy visually.

---

## 💡 Key Learnings
- Data exploration and feature selection are crucial for accurate modeling.  
- Linear Regression coefficients help interpret feature impact.  
- Visualization enhances understanding of predictions and relationships.  
- End-to-end workflow demonstrates skills in **data analysis, modeling, and evaluation**.

---

## ✅ Outcome
- Successfully built a predictive model for TotalAmount.  
- Insights into which factors (Quantity, UnitPrice, Discount) influence sales most.  
- Portfolio-ready notebook showcasing **exploratory analysis, feature importance, and predictive modeling**.

---

## 🔗 Repository Link
[View Project on GitHub](https://github.com/shifashabu/Amazon_Sales_Prediction)

