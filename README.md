# 🍽️ Smart Dining on Campus - Weekly Meal Optimizer

An interactive dashboard that creates personalized weekly meal plans using linear programming optimization.

## Features

- **Personalized Nutrition**: Customizes plans based on dietary restrictions, allergies, and health goals
- **Budget Optimization**: Minimizes costs while meeting nutritional requirements
- **Constraint Satisfaction**: Enforces 34+ constraints including variety, restaurant limits, and meal balance
- **Interactive Dashboard**: User-friendly interface with visualizations
- **Export Functionality**: Download your meal plan as CSV

## How to Use

1. Upload your meal database (CSV format)
2. Set your dietary preferences and budget
3. Generate optimized weekly meal plan
4. View results with nutritional analysis and charts
5. Export your plan

## CSV Format Required

Your CSV must include these columns:
- `Restaurant`, `Meal`, `price`
- `calories_kcal`, `protein_g`, `fat_g`, `sugar_g`
- Binary indicators (0/1): `diabetic_friendly`, `vegan`, `vegetarian`, `pescatarian`
- `contains_gluten`, `contains_lactose`, `contains_nuts`, `fried`, `grilled`, `baked`
- Additional nutritional data: `calcium_mg`, `fiber_mg`, `cholesterol_mg`, etc.

## Optimization Constraints

The system automatically applies:
- Budget limits
- Meal uniqueness (no repeats)
- Restaurant variety rules
- Nutritional balance (calories, protein, vitamins)
- Preparation method balance
- Daily rules (lunch > dinner calories, no legumes at dinner)

## Technologies Used

- **Streamlit**: Interactive web dashboard
- **PuLP**: Linear programming optimization
- **Pandas**: Data manipulation
- **Plotly**: Interactive charts

## Author

Operations Research Project - University Course

## License

Educational Use Only
