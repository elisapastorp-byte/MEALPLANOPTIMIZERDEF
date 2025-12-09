# Smart Dining Optimizer – Weekly Meal Planning Dashboard

An interactive Streamlit dashboard that generates a personalized weekly meal plan using a goal-programming optimization model. The system balances nutritional requirements, dietary restrictions, preferences, and budget constraints, creating 14 optimized meals (lunch + dinner for 7 days).

Users interact only through a clean graphical interface while the optimization runs in the background.

**Live Dashboard:**  
👉 https://mealplanoptimizerdef-apr9stfdwfffpbx7kycubn.streamlit.app/

---

## 1. Overview

The Smart Dining Optimizer is designed to demonstrate how mathematical optimization can be applied in a realistic dining-service or wellness-program setting.  
The tool automatically:

- Generates an optimal weekly schedule of meals  
- Ensures feasibility under strict nutritional and dietary constraints  
- Minimizes deviations from nutritional targets (primary goal)  
- Minimizes total weekly cost (secondary goal)  
- Presents results in business-friendly visualizations  

The dashboard is fully interactive and requires **no coding knowledge**.

---

## 2. How to Use the Dashboard

### **Step 1 — Upload a Dataset (Optional)**
Users may upload a CSV file containing meal information.  
If no file is uploaded, a validated default dataset is used.

### **Step 2 — Complete the User Profile**
Adjust all relevant inputs:

#### **Personal parameters**
- Gender  
- Weekly budget  

#### **Dietary styles**
- Vegan  
- Vegetarian  
- Pescatarian  

#### **Allergies & intolerances**
- Gluten-free  
- Lactose-free  
- Nut-free  

#### **Religious considerations**
- Kosher  
- Halal  

#### **Health & lifestyle goals**
- Keto  
- Weight loss / Weight gain  
- Muscle gain  

#### **Food preferences**
- Avoid grains  
- Avoid legumes  
- Avoid bread  
- Avoid dairy  
- Avoid spicy food  
- Avoid fried food  

Inputs use checkboxes, selectors, or numeric fields for ease of use.

---

## 3. Running the Optimization

After inputs are selected, click **“Run optimization”**.

The system will:

1. Validate and filter the dataset  
2. Apply all dietary, nutritional, and logical constraints  
3. Solve a mixed-integer goal-programming optimization model  
4. Display the weekly plan and performance metrics  

No solver output or code is shown to the user — the dashboard behaves as a standalone decision-support tool.

---

## 4. Understanding the Results

The dashboard presents the results through several sections:

### **Key Performance Indicators (KPIs)**
- Total weekly cost  
- Average daily cost  
- Average daily calories  

### **Graphs**
- **Cost per day**  
- **Calories per day**  
Both charts are displayed in chronological order: **Monday → Sunday**.

### **Weekly Meal Plan Table**
For each meal (lunch & dinner) the table shows:

- Day and meal type  
- Restaurant  
- Dish name  
- Price  
- Calories and macronutrients  
- Additional nutritional info  

The table is sorted and easy to interpret.  
A **CSV export** option allows users to download the optimized plan.

---

## 5. CSV Format Requirements

Uploaded datasets **must include** these columns:

### **Core fields**
- `Restaurant`, `Meal`, `price`
- `calories_kcal`, `protein_g`, `fat_g`, `sugar_g`

### **Binary dietary indicators (0/1)**
- `diabetic_friendly`, `vegan`, `vegetarian`, `pescatarian`
- `contains_gluten`, `contains_lactose`, `contains_nuts`
- `contains_grains`, `contains_legumes`, `contains_bread`, `contains_dairy`

### **Cooking method indicators**
- `fried`, `grilled`, `baked`, `boiled`

### **Additional nutritional fields**
- `calcium_mg`, `fiber_mg`, `cholesterol_mg`
- `potassium_mg`, `iron_mg`, `sodium_mg`

The model automatically rejects meals priced below $5 to avoid selecting side dishes.

---

## 6. Optimization Model Summary

The dashboard uses a mixed-integer programming model with:

### **Goal-Programming Objective**
1. **Primary:** minimize daily nutritional deviations  
2. **Secondary:** minimize total weekly cost  

### **Included constraints (non-exhaustive):**
- Weekly budget limit  
- Meal uniqueness (no meal repeats)  
- Restaurant variety and daily limits  
- Minimum/maximum daily nutritional targets  
- Logical constraints (e.g., no legumes at dinner, lunch > dinner calories)  
- Fried/grilled/baked/boiled meal distribution rules  
- Consecutive-day consumption limits  
- Preference-based filtering  

The model is solved using the CBC solver through PuLP.

---

## 7. Technologies Used

- **Streamlit** — user interface  
- **PuLP** — linear/mixed-integer optimization  
- **Pandas** — data handling  
- **Plotly / Streamlit Charts** — visual analytics  

---

## 8. Business Applications

The tool demonstrates how optimization can support:

- University dining services  
- Corporate wellness programs  
- Menu planning under nutritional and budgetary constraints  
- Health-focused decision support systems  

---

## 9. Links

- **Live App:** https://mealplanoptimizerdef-apr9stfdwfffpbx7kycubn.streamlit.app/  
- **Repository:** *Add your GitHub repo link here*

---
