# Smart Dining on Campus – Weekly Meal Optimizer

This project provides an interactive dashboard that generates a personalized weekly meal plan using a mixed-integer optimization model. The system balances nutritional requirements, dietary restrictions, variety constraints, and budget considerations through a goal-programming objective.

---

## Features

- **Personalized nutrition:** Incorporates allergies, dietary restrictions, and health goals.
- **Goal-programming optimization:** Prioritizes minimizing nutritional deviations while keeping total weekly cost low.
- **Over 30 operational constraints:** Includes variety rules, preparation-method limits, restaurant balance, daily nutritional bounds, and feasibility filters.
- **Interactive dashboard:** Built with Streamlit, allowing users to adjust model parameters and upload custom datasets.
- **Data validation:** Ensures the uploaded CSV meets the required schema.
- **Visual results:** Daily cost and calorie charts, KPIs, and an ordered weekly plan table.
- **Export:** Users may download the generated plan as a CSV file.

---

## How to Use

1. Upload your meal database (CSV format) or use the default dataset.
2. Select dietary preferences, allergies, nutritional goals, and weekly budget.
3. Run the optimizer to generate a 7-day meal plan (lunch + dinner).
4. Explore KPIs and daily charts.
5. Download the final plan if desired.

---

## Required CSV Format

The uploaded file must contain **all** of the following columns:

- **Basic fields:**  
  `Restaurant`, `Meal`, `price`, `calories_kcal`, `protein_g`, `fat_g`, `sugar_g`

- **Binary dietary indicators:**  
  `vegan`, `vegetarian`, `pescatarian`,  
  `diabetic_friendly`,  
  `contains_gluten`, `contains_lactose`, `contains_nuts`,  
  `contains_grains`, `contains_legumes`, `contains_bread`, `contains_dairy`

- **Cooking method flags:**  
  `fried`, `grilled`, `baked`, `boiled`

- **Additional nutritional fields:**  
  `calcium_mg`, `fiber_mg`, `cholesterol_mg`,  
  `potassium_mg`, `iron_mg`, `sodium_mg`

All binary fields should use **0/1** values.

---

## Optimization Model

The optimizer enforces a wide set of constraints, including:

- **Budget and cost control**
- **Uniqueness:** A meal cannot appear more than once in the weekly plan  
- **Restaurant variety:** Limits on how frequently each restaurant can appear
- **Dietary restrictions and allergens**
- **Daily nutritional targets:** Calories, macronutrients, and selected micronutrients
- **Multiple logical rules:**  


### Objective Function

A **goal-programming approach** is used:

1. Primary goal: minimize deviations from nutritional targets.  
2. Secondary goal: minimize total weekly cost.  

This ensures feasible and balanced plans while remaining budget-efficient.

---

## Technologies Used

- **Streamlit** — interactive web application  
- **PuLP** — mixed-integer optimization  
- **Pandas** — data cleaning and manipulation  
- **Plotly / Streamlit charts** — visualizations  

---

