import pandas as pd
import pulp as pl
import streamlit as st

# Columns required in the CSV (exact names, including spaces and suffixes)
REQUIRED_COLUMNS = [
    'Restaurant', 'Meal', 'price', 'calories_kcal', 'protein_g', 'fat_g',
    'sugar_g', 'contains_gluten', 'contains_lactose', 'diabetic_friendly',
    'vegan ', 'vegetarian', 'pescatarian', 'kosher', 'halal', 'contains_nuts',
    'contains_lactose.1', 'carbs_g', 'calcium_mg', 'fiber_mg',
    'cholesterol_mg', 'potassium_mg', 'iron_mg', 'sodium_mg',
    'contains_grains', 'contains_legumes', 'contains_bread', 'contains_dairy',
    'keto_friendly', 'gaining_weight_diet', 'loose_weight_diet',
    'gaining_muscle_diet', 'spicy', 'fried', 'grilled ', 'baked', 'boiled'
]

DAY_NAMES = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}

MEAL_TYPES = ["lunch", "dinner"]


def load_meal_data(uploaded_file):
    """
    Load CSV either from user upload or from the default file.
    Validate that all required columns are present.
    """
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        source = "uploaded file"
    else:
        df = pd.read_csv("lunchplandef3.csv")
        source = "default file (lunchplandef3.csv)"

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "The selected dataset does not have the required structure.\n"
            f"Missing columns: {missing}\n"
            "Please make sure your CSV contains all required columns."
        )

    return df, source


def apply_filters(df, gender, prefs):
    """Apply dietary filters based on user preferences."""
    filtered = df.copy()

    # Allergies / intolerances
    if prefs.get("celiac"):
        filtered = filtered[filtered["contains_gluten"] == 0]
    if prefs.get("lactose_intolerant"):
        filtered = filtered[filtered["contains_lactose"] == 0]
    if prefs.get("nut_allergy"):
        filtered = filtered[filtered["contains_nuts"] == 0]

    # Diet type
    # NOTE: vegan column has a space in the name: 'vegan '
    if prefs.get("vegan"):
        filtered = filtered[filtered["vegan "] == 1]
    if prefs.get("vegetarian"):
        filtered = filtered[filtered["vegetarian"] == 1]
    if prefs.get("pescatarian"):
        filtered = filtered[filtered["pescatarian"] == 1]

    # Religious
    if prefs.get("kosher"):
        filtered = filtered[filtered["kosher"] == 1]
    if prefs.get("halal"):
        filtered = filtered[filtered["halal"] == 1]

    # Diet goals
    if prefs.get("keto"):
        filtered = filtered[filtered["keto_friendly"] == 1]
    if prefs.get("gain_weight"):
        filtered = filtered[filtered["gaining_weight_diet"] == 1]
    if prefs.get("lose_weight"):
        filtered = filtered[filtered["loose_weight_diet"] == 1]
    if prefs.get("gain_muscle"):
        filtered = filtered[filtered["gaining_muscle_diet"] == 1]

    # Content preferences
    if prefs.get("avoid_bread"):
        filtered = filtered[filtered["contains_bread"] == 0]
    if prefs.get("avoid_grains"):
        filtered = filtered[filtered["contains_grains"] == 0]
    if prefs.get("avoid_legumes"):
        filtered = filtered[filtered["contains_legumes"] == 0]
    if prefs.get("avoid_dairy"):
        filtered = filtered[filtered["contains_dairy"] == 0]
    if prefs.get("avoid_spicy"):
        filtered = filtered[filtered["spicy"] == 0]
    if prefs.get("avoid_fried"):
        filtered = filtered[filtered["fried"] == 0]

    # Nutritional targets by gender (per day) – simplified
    if gender == "male":
        bounds = dict(
            cal_min=1100,
            cal_max=1600,
            protein_min=50,
            fat_max=70,
            sugar_max=70,
        )
    else:  # female
        bounds = dict(
            cal_min=900,
            cal_max=1400,
            protein_min=45,
            fat_max=65,
            sugar_max=60,
        )

    return filtered.reset_index(drop=True), bounds


def optimize_weekly_plan(filtered, weekly_budget, bounds):
    """Run a simple min-cost optimization with daily nutritional bounds."""
    n_days = 7
    days = range(n_days)

    if len(filtered) < n_days * len(MEAL_TYPES):
        return None, "Not enough meals after filtering."

    meal_indices = list(filtered.index)

    prob = pl.LpProblem("SmartDiningWeeklyPlan", pl.LpMinimize)

    # Decision variables x[i, d, m] in {0,1}
    x = {}
    for i in meal_indices:
        for d in days:
            for m in MEAL_TYPES:
                x[(i, d, m)] = pl.LpVariable(f"x_{i}_{d}_{m}", cat="Binary")

    # 1) Exactly one meal per day and meal type
    for d in days:
        for m in MEAL_TYPES:
            prob += (
                pl.lpSum(x[(i, d, m)] for i in meal_indices) == 1,
                f"OneMeal_day{d}_{m}",
            )

    # 2) Weekly budget constraint
    total_cost = pl.lpSum(
        filtered.loc[i, "price"] * x[(i, d, m)]
        for i in meal_indices
        for d in days
        for m in MEAL_TYPES
    )
    prob += total_cost <= weekly_budget, "WeeklyBudget"

    # 3) Daily nutritional bounds (simplified)
    for d in days:
        calories_day = pl.lpSum(
            filtered.loc[i, "calories_kcal"] * x[(i, d, m)]
            for i in meal_indices
            for m in MEAL_TYPES
        )
        protein_day = pl.lpSum(
            filtered.loc[i, "protein_g"] * x[(i, d, m)]
            for i in meal_indices
            for m in MEAL_TYPES
        )
        fat_day = pl.lpSum(
            filtered.loc[i, "fat_g"] * x[(i, d, m)]
            for i in meal_indices
            for m in MEAL_TYPES
        )
        sugar_day = pl.lpSum(
            filtered.loc[i, "sugar_g"] * x[(i, d, m)]
            for i in meal_indices
            for m in MEAL_TYPES
        )

        prob += calories_day >= bounds["cal_min"], f"CalMin_day{d}"
        prob += calories_day <= bounds["cal_max"], f"CalMax_day{d}"
        prob += protein_day >= bounds["protein_min"], f"ProtMin_day{d}"
        prob += fat_day <= bounds["fat_max"], f"FatMax_day{d}"
        prob += sugar_day <= bounds["sugar_max"], f"SugarMax_day{d}"

    # Objective: minimize total weekly cost
    prob += total_cost

    solver = pl.PULP_CBC_CMD(msg=False, timeLimit=60, gapRel=0.01)
    prob.solve(solver)

    status = pl.LpStatus[prob.status]
    if status != "Optimal":
        return None, f"No optimal solution found. Status: {status}"

    # Build chosen plan dataframe
    chosen_rows = []
    for d in days:
        for m in MEAL_TYPES:
            for i in meal_indices:
                if pl.value(x[(i, d, m)]) > 0.5:
                    row = filtered.loc[i].copy()
                    row["day"] = d
                    row["day_name"] = DAY_NAMES[d]
                    row["meal_type"] = m
                    chosen_rows.append(row)

    plan = pd.DataFrame(chosen_rows)
    if plan.empty:
        return None, "Solver returned no chosen meals."

    # KPIs
    kpis = {
        "total_cost": plan["price"].sum(),
        "avg_daily_cost": plan.groupby("day")["price"].sum().mean(),
        "avg_daily_calories": plan.groupby("day")["calories_kcal"].sum().mean(),
    }

    return {"plan": plan, "kpis": kpis, "status": status}, None


def main():
    st.set_page_config(
        page_title="Smart Dining on Campus",
        layout="wide",
    )

    st.title("Smart Dining on Campus - Weekly Planner")
    st.markdown(
        "This dashboard builds a **7-day lunch & dinner plan** subject to "
        "your dietary preferences, nutrition targets, and budget."
    )

    # Sidebar: inputs
    st.sidebar.header("User & Preferences")

    st.sidebar.write(
        "You can upload your own **CSV** file with meal data. "
        "If you do not upload any file, the default dataset "
        "`lunchplandef3.csv` will be used."
    )

    with st.sidebar.expander("Required CSV structure", expanded=False):
        st.markdown(
            "**The CSV must contain at least these columns (exact names):**\n\n"
            ", ".join(f"`{c}`" for c in REQUIRED_COLUMNS)
        )

    uploaded_file = st.sidebar.file_uploader(
        "Upload meal data (CSV)",
        type=["csv"],
        help="If empty, the default dataset 'lunchplandef3.csv' will be used.",
    )

    gender = st.sidebar.selectbox("Gender (for nutritional targets)", ["female", "male"])

    weekly_budget = st.sidebar.number_input(
        "Maximum weekly budget (USD)",
        min_value=20.0,
        max_value=300.0,
        value=120.0,
        step=5.0,
    )

    st.sidebar.subheader("Allergies / intolerances")
    celiac = st.sidebar.checkbox("Celiac (gluten-free)", value=False)
    lactose_intolerant = st.sidebar.checkbox("Lactose intolerant", value=False)
    nut_allergy = st.sidebar.checkbox("Nut allergy", value=False)

    st.sidebar.subheader("Diet type")
    vegan = st.sidebar.checkbox("Vegan", value=False)
    vegetarian = st.sidebar.checkbox("Vegetarian", value=False)
    pescatarian = st.sidebar.checkbox("Pescatarian", value=False)

    st.sidebar.subheader("Religious preferences")
    kosher = st.sidebar.checkbox("Kosher", value=False)
    halal = st.sidebar.checkbox("Halal", value=False)

    st.sidebar.subheader("Diet goals")
    keto = st.sidebar.checkbox("Keto", value=False)
    gain_weight = st.sidebar.checkbox("Gain weight", value=False)
    lose_weight = st.sidebar.checkbox("Lose weight", value=False)
    gain_muscle = st.sidebar.checkbox("Gain muscle", value=False)

    st.sidebar.subheader("Content preferences")
    avoid_bread = st.sidebar.checkbox("Avoid bread", value=False)
    avoid_grains = st.sidebar.checkbox("Avoid grains", value=False)
    avoid_legumes = st.sidebar.checkbox("Avoid legumes", value=False)
    avoid_dairy = st.sidebar.checkbox("Avoid dairy", value=False)
    avoid_spicy = st.sidebar.checkbox("Avoid spicy", value=False)
    avoid_fried = st.sidebar.checkbox("Avoid fried food", value=False)

    prefs = {
        "celiac": celiac,
        "lactose_intolerant": lactose_intolerant,
        "nut_allergy": nut_allergy,
        "vegan": vegan,
        "vegetarian": vegetarian,
        "pescatarian": pescatarian,
        "kosher": kosher,
        "halal": halal,
        "keto": keto,
        "gain_weight": gain_weight,
        "lose_weight": lose_weight,
        "gain_muscle": gain_muscle,
        "avoid_bread": avoid_bread,
        "avoid_grains": avoid_grains,
        "avoid_legumes": avoid_legumes,
        "avoid_dairy": avoid_dairy,
        "avoid_spicy": avoid_spicy,
        "avoid_fried": avoid_fried,
    }

    if st.sidebar.button("Run optimization"):
        with st.spinner("Running optimization model..."):
            try:
                df, source = load_meal_data(uploaded_file)
            except Exception as e:
                st.error(str(e))
                st.stop()

            st.info(f"Using data source: **{source}**")
            filtered, bounds = apply_filters(df, gender, prefs)

            st.write(f"Filtered meals available: **{len(filtered)}**")

            result, error = optimize_weekly_plan(filtered, weekly_budget, bounds)

        if error:
            st.error(error)
        else:
            plan = result["plan"]
            kpis = result["kpis"]

            st.success(f"Optimization completed. Status: {result['status']}")

            # KPIs
            st.subheader("Key Performance Indicators")
            kpi_cols = st.columns(3)
            kpi_cols[0].metric("Total weekly cost (USD)", f"{kpis['total_cost']:.2f}")
            kpi_cols[1].metric("Avg daily cost (USD)", f"{kpis['avg_daily_cost']:.2f}")
            kpi_cols[2].metric(
                "Avg daily calories (kcal)", f"{kpis['avg_daily_calories']:.0f}"
            )

            # Detailed plan
            st.subheader("Weekly meal plan")
            st.dataframe(
                plan[
                    [
                        "day_name",
                        "meal_type",
                        "Restaurant",
                        "Meal",
                        "price",
                        "calories_kcal",
                        "protein_g",
                        "fat_g",
                        "sugar_g",
                    ]
                ].sort_values(["day", "meal_type"])
            )

            # Charts
            st.subheader("Cost per day")
            cost_per_day = (
                plan.groupby("day_name")["price"].sum().reindex(DAY_NAMES.values())
            )
            st.bar_chart(cost_per_day)

            st.subheader("Calories per day")
            cal_per_day = (
                plan.groupby("day_name")["calories_kcal"].sum().reindex(DAY_NAMES.values())
            )
            st.bar_chart(cal_per_day)

    else:
        st.info(
            "Upload a CSV file (optional), set your preferences and click "
            "**Run optimization** to see a weekly plan."
        )


if __name__ == "__main__":
    main()
