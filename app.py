import pandas as pd
import pulp as pl
import streamlit as st

# -----------------------------
# Basic configuration
# -----------------------------

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

REQUIRED_COLUMNS = [
    "Restaurant", "Meal", "price",
    "calories_kcal", "protein_g", "fat_g", "sugar_g",
    "contains_gluten", "contains_lactose", "contains_nuts",
    "diabetic_friendly",
    "vegan", "vegetarian", "pescatarian",
    "kosher", "halal",
    "contains_grains", "contains_legumes", "contains_bread", "contains_dairy",
    "keto_friendly", "gaining_weight_diet", "loose_weight_diet", "gaining_muscle_diet",
    "spicy", "fried", "grilled", "baked", "boiled",
    "calcium_mg", "fiber_mg", "cholesterol_mg",
    "potassium_mg", "iron_mg", "sodium_mg",
]


# -----------------------------
# Data loading
# -----------------------------

def load_meal_data(uploaded_file):
    """
    Load the meal dataset either from a user-uploaded CSV
    or from the default 'lunchplandef3.csv' file.
    Column names are trimmed and validated.
    """
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        source = "uploaded file"
    else:
        df = pd.read_csv("lunchplandef3.csv")
        source = "default file (lunchplandef3.csv)"

    # strip leading/trailing spaces in column names
    df = df.rename(columns={c: c.strip() for c in df.columns})

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Dataset does not contain the required columns.\n"
            f"Missing columns: {missing}"
        )

    return df, source


# -----------------------------
# Optimization model
# -----------------------------

def solve_smart_dining(df, gender, weekly_budget, prefs):
    """
    Build and solve the weekly meal plan optimization model
    using a goal-programming objective.
    """

    # User options
    q_diabetic      = prefs.get("diabetic", False)
    q_vegan         = prefs.get("vegan", False)
    q_vegetarian    = prefs.get("vegetarian", False)
    q_pescatarian   = prefs.get("pescatarian", False)
    q_celiac        = prefs.get("celiac", False)
    q_lactose       = prefs.get("lactose_intolerant", False)
    q_nut_allergy   = prefs.get("nut_allergy", False)
    q_kosher        = prefs.get("kosher", False)
    q_halal         = prefs.get("halal", False)
    q_keto          = prefs.get("keto", False)
    q_gain_weight   = prefs.get("gain_weight", False)
    q_lose_weight   = prefs.get("lose_weight", False)
    q_gain_muscle   = prefs.get("gain_muscle", False)
    q_avoid_grains  = prefs.get("avoid_grains", False)
    q_avoid_legumes = prefs.get("avoid_legumes", False)
    q_avoid_bread   = prefs.get("avoid_bread", False)
    q_avoid_dairy   = prefs.get("avoid_dairy", False)
    q_avoid_spicy   = prefs.get("avoid_spicy", False)
    q_avoid_fried   = prefs.get("avoid_fried", False)

    # 1. Filter dataset based on user preferences
    filtered = df.copy()
    if q_diabetic:
        filtered = filtered[filtered["diabetic_friendly"] == 1]
    if q_vegan:
        filtered = filtered[filtered["vegan"] == 1]
    if q_vegetarian:
        filtered = filtered[filtered["vegetarian"] == 1]
    if q_pescatarian:
        filtered = filtered[filtered["pescatarian"] == 1]
    if q_celiac:
        filtered = filtered[filtered["contains_gluten"] == 0]
    if q_lactose:
        filtered = filtered[filtered["contains_lactose"] == 0]
    if q_nut_allergy:
        filtered = filtered[filtered["contains_nuts"] == 0]
    if q_kosher:
        filtered = filtered[filtered["kosher"] == 1]
    if q_halal:
        filtered = filtered[filtered["halal"] == 1]
    if q_keto:
        filtered = filtered[filtered["keto_friendly"] == 1]
    if q_gain_weight:
        filtered = filtered[filtered["gaining_weight_diet"] == 1]
    if q_lose_weight:
        filtered = filtered[filtered["loose_weight_diet"] == 1]
    if q_gain_muscle:
        filtered = filtered[filtered["gaining_muscle_diet"] == 1]
    if q_avoid_grains:
        filtered = filtered[filtered["contains_grains"] == 0]
    if q_avoid_legumes:
        filtered = filtered[filtered["contains_legumes"] == 0]
    if q_avoid_bread:
        filtered = filtered[filtered["contains_bread"] == 0]
    if q_avoid_dairy:
        filtered = filtered[filtered["contains_dairy"] == 0]
    if q_avoid_spicy:
        filtered = filtered[filtered["spicy"] == 0]
    if q_avoid_fried:
        filtered = filtered[filtered["fried"] == 0]

    if len(filtered) < 14:
        return None, f"Not enough meals after filtering. Only {len(filtered)} meals available."

    # 2. Daily nutritional targets by gender
    if gender == "male":
        cal_min       = 1100
        cal_max       = 1600
        protein_min   = 50
        fat_max       = 70
        sugar_max     = 70
        calcium_min   = 900
        fiber_min     = 30
        chol_max      = 300
        potassium_min = 3400
        iron_min      = 8
        sodium_max    = 2300
    else:  # female
        cal_min       = 900
        cal_max       = 1400
        protein_min   = 45
        fat_max       = 60
        sugar_max     = 60
        calcium_min   = 1200
        fiber_min     = 25
        chol_max      = 300
        potassium_min = 2600
        iron_min      = 18
        sodium_max    = 2300

    # 3. Sets and indices
    meal_indices = list(filtered.index)
    days = range(7)
    meals = MEAL_TYPES
    restaurants = filtered["Restaurant"].unique().tolist()

    prob = pl.LpProblem("SmartDiningGoalProgramming", pl.LpMinimize)

    # Decision variables: x[i,d,m] = 1 if meal i is chosen on day d, meal m
    x = {}
    for i in meal_indices:
        for d in days:
            for m in meals:
                x[(i, d, m)] = pl.LpVariable(f"x_{i}_{d}_{m}", cat="Binary")

    # Total weekly cost
    total_cost = pl.lpSum(
        filtered.loc[i, "price"] * x[(i, d, m)]
        for i in meal_indices
        for d in days
        for m in meals
    )

    # Weekly budget as a hard constraint
    prob += total_cost <= weekly_budget, "B1_BudgetMaxWeeklyCost"

    # 4. Goal-programming deviation variables
    d_cal_under   = {d: pl.LpVariable(f"d_cal_under_day{d}",    lowBound=0) for d in days}
    d_cal_over    = {d: pl.LpVariable(f"d_cal_over_day{d}",     lowBound=0) for d in days}
    d_prot_under  = {d: pl.LpVariable(f"d_prot_under_day{d}",   lowBound=0) for d in days}
    d_fat_over    = {d: pl.LpVariable(f"d_fat_over_day{d}",     lowBound=0) for d in days}
    d_sugar_over  = {d: pl.LpVariable(f"d_sugar_over_day{d}",   lowBound=0) for d in days}
    d_calc_under  = {d: pl.LpVariable(f"d_calcium_under_day{d}",lowBound=0) for d in days}
    d_fiber_under = {d: pl.LpVariable(f"d_fiber_under_day{d}",  lowBound=0) for d in days}
    d_chol_over   = {d: pl.LpVariable(f"d_chol_over_day{d}",    lowBound=0) for d in days}
    d_potas_under = {d: pl.LpVariable(f"d_potas_under_day{d}",  lowBound=0) for d in days}
    d_iron_under  = {d: pl.LpVariable(f"d_iron_under_day{d}",   lowBound=0) for d in days}
    d_sodium_over = {d: pl.LpVariable(f"d_sodium_over_day{d}",  lowBound=0) for d in days}

    # Daily nutritional soft constraints
    for d in days:
        calories_day = pl.lpSum(
            filtered.loc[i, "calories_kcal"] * x[(i, d, m)]
            for i in meal_indices for m in meals
        )
        protein_day = pl.lpSum(
            filtered.loc[i, "protein_g"] * x[(i, d, m)]
            for i in meal_indices for m in meals
        )
        fat_day = pl.lpSum(
            filtered.loc[i, "fat_g"] * x[(i, d, m)]
            for i in meal_indices for m in meals
        )
        sugar_day = pl.lpSum(
            filtered.loc[i, "sugar_g"] * x[(i, d, m)]
            for i in meal_indices for m in meals
        )
        calcium_day = pl.lpSum(
            filtered.loc[i, "calcium_mg"] * x[(i, d, m)]
            for i in meal_indices for m in meals
        )
        fiber_day = pl.lpSum(
            filtered.loc[i, "fiber_mg"] * x[(i, d, m)]
            for i in meal_indices for m in meals
        )
        chol_day = pl.lpSum(
            filtered.loc[i, "cholesterol_mg"] * x[(i, d, m)]
            for i in meal_indices for m in meals
        )
        potassium_day = pl.lpSum(
            filtered.loc[i, "potassium_mg"] * x[(i, d, m)]
            for i in meal_indices for m in meals
        )
        iron_day = pl.lpSum(
            filtered.loc[i, "iron_mg"] * x[(i, d, m)]
            for i in meal_indices for m in meals
        )
        sodium_day = pl.lpSum(
            filtered.loc[i, "sodium_mg"] * x[(i, d, m)]
            for i in meal_indices for m in meals
        )

        prob += calories_day + d_cal_under[d]    >= cal_min,       f"C6_CalMinSoft_day{d}"
        prob += calories_day - d_cal_over[d]     <= cal_max,       f"C7_CalMaxSoft_day{d}"
        prob += protein_day + d_prot_under[d]    >= protein_min,   f"C8_ProtMinSoft_day{d}"
        prob += fat_day - d_fat_over[d]          <= fat_max,       f"C9_FatMaxSoft_day{d}"
        prob += sugar_day - d_sugar_over[d]      <= sugar_max,     f"C10_SugarMaxSoft_day{d}"
        prob += calcium_day + d_calc_under[d]    >= calcium_min,   f"C11_CalciumMinSoft_day{d}"
        prob += fiber_day + d_fiber_under[d]     >= fiber_min,     f"C12_FiberMinSoft_day{d}"
        prob += chol_day - d_chol_over[d]        <= chol_max,      f"C13_CholMaxSoft_day{d}"
        prob += potassium_day + d_potas_under[d] >= potassium_min, f"C14_PotassiumMinSoft_day{d}"
        prob += iron_day + d_iron_under[d]       >= iron_min,      f"C15_IronMinSoft_day{d}"
        prob += sodium_day - d_sodium_over[d]    <= sodium_max,    f"C16_SodiumMaxSoft_day{d}"

    # 5. Combinatorial constraints

    # Each day and meal type: exactly one meal
    for d in days:
        for m in meals:
            prob += pl.lpSum(
                x[(i, d, m)] for i in meal_indices
            ) == 1, f"C1_OneMeal_day{d}_{m}"

    # Each meal at most once per week
    for i in meal_indices:
        prob += pl.lpSum(
            x[(i, d, m)] for d in days for m in meals
        ) <= 1, f"C2_UniqueMeal_{i}"

    # Restaurant usage constraints
    for r in restaurants:
        # Max 5 meals from same restaurant per week
        prob += pl.lpSum(
            x[(i, d, m)]
            for i in meal_indices
            for d in days
            for m in meals
            if filtered.loc[i, "Restaurant"] == r
        ) <= 5, f"C3_Max5Meals_rest_{r}"

    # Max 1 meal from same restaurant per day
    for d in days:
        for r in restaurants:
            prob += pl.lpSum(
                x[(i, d, m)]
                for i in meal_indices
                for m in meals
                if filtered.loc[i, "Restaurant"] == r
            ) <= 1, f"C4_MaxOnePerRestaurant_day{d}_rest_{r}"

    # Avoid same restaurant on consecutive days for the same meal type
    for d in range(len(days) - 1):
        for r in restaurants:
            for m in meals:
                prob += (
                    pl.lpSum(
                        x[(i, d, m)]
                        for i in meal_indices
                        if filtered.loc[i, "Restaurant"] == r
                    )
                    +
                    pl.lpSum(
                        x[(i, d + 1, m)]
                        for i in meal_indices
                        if filtered.loc[i, "Restaurant"] == r
                    )
                ) <= 1, f"C5_NoSameRestConsecutive_{m}_day{d}_rest_{r}"

    # Weekly content constraints
    prob += pl.lpSum(
        x[(i, d, m)]
        for i in meal_indices for d in days for m in meals
        if filtered.loc[i, "fried"] == 1
    ) <= 2, "C17_MaxFriedWeek"

    prob += pl.lpSum(
        x[(i, d, m)]
        for i in meal_indices for d in days for m in meals
        if filtered.loc[i, "grilled"] == 1
    ) >= 3, "C18_MinGrilledWeek"

    prob += pl.lpSum(
        x[(i, d, m)]
        for i in meal_indices for d in days for m in meals
        if filtered.loc[i, "baked"] == 1
    ) >= 2, "C19_MinBakedWeek"

    prob += pl.lpSum(
        x[(i, d, m)]
        for i in meal_indices for d in days for m in meals
        if filtered.loc[i, "boiled"] == 1
    ) >= 1, "C20_MinBoiledWeek"

    prob += pl.lpSum(
        x[(i, d, m)]
        for i in meal_indices for d in days for m in meals
        if filtered.loc[i, "contains_legumes"] == 1
    ) >= 2, "C21_MinLegumesWeek"

    prob += pl.lpSum(
        x[(i, d, m)]
        for i in meal_indices for d in days for m in meals
        if filtered.loc[i, "contains_bread"] == 1
    ) <= 5, "C22_MaxBreadWeek"

    prob += pl.lpSum(
        x[(i, d, m)]
        for i in meal_indices for d in days for m in meals
        if filtered.loc[i, "contains_grains"] == 0
    ) >= 1, "C23_MinGrainFreeWeek"

    # Daily constraints on fried and composition
    for d in days:
        # Max 1 fried meal per day
        prob += pl.lpSum(
            x[(i, d, m)]
            for i in meal_indices for m in meals
            if filtered.loc[i, "fried"] == 1
        ) <= 1, f"C24_MaxFriedPerDay_{d}"

        # At least one high-protein meal per day (protein >= 25g)
        prob += pl.lpSum(
            x[(i, d, m)]
            for i in meal_indices for m in meals
            if filtered.loc[i, "protein_g"] >= 25
        ) >= 1, f"C25_MinHighProteinPerDay_{d}"

        # Lunch calories >= dinner calories
        prob += pl.lpSum(
            filtered.loc[i, "calories_kcal"] * x[(i, d, "lunch")]
            for i in meal_indices
        ) >= pl.lpSum(
            filtered.loc[i, "calories_kcal"] * x[(i, d, "dinner")]
            for i in meal_indices
        ), f"C26_LunchMoreCalories_{d}"

        # No legumes at dinner
        prob += pl.lpSum(
            x[(i, d, "dinner")]
            for i in meal_indices
            if filtered.loc[i, "contains_legumes"] == 1
        ) == 0, f"C27_NoLegumesDinner_{d}"

        # No grains at dinner
        prob += pl.lpSum(
            x[(i, d, "dinner")]
            for i in meal_indices
            if filtered.loc[i, "contains_grains"] == 1
        ) == 0, f"C28_NoGrainsDinner_{d}"

    # Consecutive-day patterns
    for idx in range(len(days) - 1):
        d = days[idx]
        d_next = days[idx + 1]

        # No legumes on two consecutive days
        prob += (
            pl.lpSum(
                x[(i, d, m)]
                for i in meal_indices for m in meals
                if filtered.loc[i, "contains_legumes"] == 1
            )
            +
            pl.lpSum(
                x[(i, d_next, m)]
                for i in meal_indices for m in meals
                if filtered.loc[i, "contains_legumes"] == 1
            )
        ) <= 1, f"C29_NoConsecutiveLegumes_{d}_{d_next}"

        # No fried lunch on two consecutive days
        prob += (
            pl.lpSum(
                x[(i, d, "lunch")]
                for i in meal_indices
                if filtered.loc[i, "fried"] == 1
            )
            +
            pl.lpSum(
                x[(i, d_next, "lunch")]
                for i in meal_indices
                if filtered.loc[i, "fried"] == 1
            )
        ) <= 1, f"C30_NoConsecutiveFriedLunch_{d}_{d_next}"

        # No fried dinner on two consecutive days
        prob += (
            pl.lpSum(
                x[(i, d, "dinner")]
                for i in meal_indices
                if filtered.loc[i, "fried"] == 1
            )
            +
            pl.lpSum(
                x[(i, d_next, "dinner")]
                for i in meal_indices
                if filtered.loc[i, "fried"] == 1
            )
        ) <= 1, f"C31_NoConsecutiveFriedDinner_{d}_{d_next}"

    # Conditional constraints (IF-THEN and EITHER-OR)
    BIG_M = 1000

    # If a day includes fried food, total protein that day must be high
    z_fried = {d: pl.LpVariable(f"z_fried_day{d}", cat="Binary") for d in days}
    HIGH_PROTEIN_IF_FRIED = 60

    for d in days:
        fried_day = pl.lpSum(
            x[(i, d, m)]
            for i in meal_indices for m in meals
            if filtered.loc[i, "fried"] == 1
        )
        protein_day = pl.lpSum(
            filtered.loc[i, "protein_g"] * x[(i, d, m)]
            for i in meal_indices for m in meals
        )

        prob += fried_day >= z_fried[d],             f"C32_FriedLinkLB_day{d}"
        prob += fried_day <= BIG_M * z_fried[d],      f"C32_FriedLinkUB_day{d}"
        prob += protein_day >= HIGH_PROTEIN_IF_FRIED - BIG_M * (1 - z_fried[d]), \
                f"C32_IfFriedThenHighProtein_day{d}"

    # Either low-calorie or high-protein day
    y_balance = {d: pl.LpVariable(f"y_balance_day{d}", cat="Binary") for d in days}
    LOW_CAL_MAX = cal_max - 200
    HIGH_PROTEIN_MIN = protein_min + 10

    for d in days:
        calories_day = pl.lpSum(
            filtered.loc[i, "calories_kcal"] * x[(i, d, m)]
            for i in meal_indices for m in meals
        )
        protein_day = pl.lpSum(
            filtered.loc[i, "protein_g"] * x[(i, d, m)]
            for i in meal_indices for m in meals
        )

        prob += calories_day <= LOW_CAL_MAX + BIG_M * y_balance[d], \
                f"C33_EitherLowCal_day{d}"
        prob += protein_day >= HIGH_PROTEIN_MIN - BIG_M * (1 - y_balance[d]), \
                f"C33_OrHighProtein_day{d}"

    # Either low-fat or low-sodium day
    y_fat_sodium = {d: pl.LpVariable(f"y_fat_sodium_day{d}", cat="Binary") for d in days}
    FAT_LOW_MAX = fat_max - 10
    SODIUM_LOW_MAX = sodium_max - 400

    for d in days:
        fat_day = pl.lpSum(
            filtered.loc[i, "fat_g"] * x[(i, d, m)]
            for i in meal_indices for m in meals
        )
        sodium_day = pl.lpSum(
            filtered.loc[i, "sodium_mg"] * x[(i, d, m)]
            for i in meal_indices for m in meals
        )

        prob += fat_day <= FAT_LOW_MAX + BIG_M * y_fat_sodium[d], \
                f"C34_EitherLowFat_day{d}"
        prob += sodium_day <= SODIUM_LOW_MAX + BIG_M * (1 - y_fat_sodium[d]), \
                f"C34_OrLowSodium_day{d}"

    # 6. Goal-programming objective: deviations have higher weight than cost
    total_deviation = pl.lpSum(
        d_cal_under[d]  + d_cal_over[d]  +
        d_prot_under[d] + d_fat_over[d]  +
        d_sugar_over[d] + d_calc_under[d] +
        d_fiber_under[d]+ d_chol_over[d] +
        d_potas_under[d]+ d_iron_under[d] +
        d_sodium_over[d]
        for d in days
    )

    W_NUTRITION = 10
    W_COST = 1
    prob += W_NUTRITION * total_deviation + W_COST * total_cost, "GoalObjective"

    # 7. Solve model
    solver = pl.PULP_CBC_CMD(msg=False, timeLimit=60, gapRel=0.01)
    prob.solve(solver)

    status = pl.LpStatus[prob.status]
    if status != "Optimal":
        return None, f"No optimal solution found. Solver status: {status}"

    # Build solution DataFrame
    chosen_rows = []
    for d in days:
        for m in meals:
            for i in meal_indices:
                if pl.value(x[(i, d, m)]) > 0.5:
                    row = filtered.loc[i].copy()
                    row["day"] = d
                    row["day_name"] = DAY_NAMES[d]
                    row["meal_type"] = m
                    chosen_rows.append(row)

    plan = pd.DataFrame(chosen_rows)
    if plan.empty:
        return None, "Solver returned an empty plan."

    kpis = {
        "total_cost": plan["price"].sum(),
        "avg_daily_cost": plan.groupby("day")["price"].sum().mean(),
        "avg_daily_calories": plan.groupby("day")["calories_kcal"].sum().mean(),
    }

    return {"plan": plan, "kpis": kpis, "status": status}, None


# -----------------------------
# Streamlit app
# -----------------------------

def main():
    st.set_page_config(
        page_title="Smart Dining on Campus",
        layout="wide",
    )

    st.title("Smart Dining on Campus – Weekly Planner")
    st.markdown(
        "This dashboard builds a 7-day lunch and dinner plan based on a "
        "mixed-integer optimization model with a goal-programming objective. "
        "Nutritional deviations are penalized more heavily than total cost."
    )

    # Sidebar inputs
    st.sidebar.header("User & Preferences")

    st.sidebar.write(
        "You can upload your own CSV file with meal data. "
        "If you do not upload any file, the default dataset "
        "`lunchplandef3.csv` will be used."
    )

    with st.sidebar.expander("Required CSV structure", expanded=False):
        st.markdown(
            "The CSV must contain at least these columns (exact names):\n\n"
            ", ".join(f"`{c}`" for c in REQUIRED_COLUMNS)
        )

    uploaded_file = st.sidebar.file_uploader(
        "Upload meal data (CSV)",
        type=["csv"],
        help="If empty, the default dataset 'lunchplandef3.csv' will be used.",
    )

    gender = st.sidebar.selectbox(
        "Gender (for nutritional targets)",
        ["female", "male"],
    )

    weekly_budget = st.sidebar.number_input(
        "Maximum weekly budget (USD)",
        min_value=20.0,
        max_value=300.0,
        value=150.0,
        step=5.0,
    )

    # Preferences
    st.sidebar.subheader("Health / conditions")
    diabetic = st.sidebar.checkbox("Diabetic (diabetic-friendly meals)", value=False)

    st.sidebar.subheader("Diet type")
    vegan = st.sidebar.checkbox("Vegan", value=False)
    vegetarian = st.sidebar.checkbox("Vegetarian", value=False)
    pescatarian = st.sidebar.checkbox("Pescatarian", value=False)

    st.sidebar.subheader("Allergies / intolerances")
    celiac = st.sidebar.checkbox("Celiac (gluten-free)", value=False)
    lactose_intolerant = st.sidebar.checkbox("Lactose intolerant", value=False)
    nut_allergy = st.sidebar.checkbox("Nut allergy", value=False)

    st.sidebar.subheader("Religious preferences")
    kosher = st.sidebar.checkbox("Kosher", value=False)
    halal = st.sidebar.checkbox("Halal", value=False)

    st.sidebar.subheader("Diet goals")
    keto = st.sidebar.checkbox("Keto", value=False)
    gain_weight = st.sidebar.checkbox("Gain weight", value=False)
    lose_weight = st.sidebar.checkbox("Lose weight", value=False)
    gain_muscle = st.sidebar.checkbox("Gain muscle", value=False)

    st.sidebar.subheader("Content preferences")
    avoid_grains = st.sidebar.checkbox("Avoid grains", value=False)
    avoid_legumes = st.sidebar.checkbox("Avoid legumes", value=False)
    avoid_bread = st.sidebar.checkbox("Avoid bread", value=False)
    avoid_dairy = st.sidebar.checkbox("Avoid dairy", value=False)
    avoid_spicy = st.sidebar.checkbox("Avoid spicy food", value=False)
    avoid_fried = st.sidebar.checkbox("Avoid fried food", value=False)

    prefs = {
        "diabetic": diabetic,
        "vegan": vegan,
        "vegetarian": vegetarian,
        "pescatarian": pescatarian,
        "celiac": celiac,
        "lactose_intolerant": lactose_intolerant,
        "nut_allergy": nut_allergy,
        "kosher": kosher,
        "halal": halal,
        "keto": keto,
        "gain_weight": gain_weight,
        "lose_weight": lose_weight,
        "gain_muscle": gain_muscle,
        "avoid_grains": avoid_grains,
        "avoid_legumes": avoid_legumes,
        "avoid_bread": avoid_bread,
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

            result, error = solve_smart_dining(
                df=df,
                gender=gender,
                weekly_budget=weekly_budget,
                prefs=prefs,
            )

        if error:
            st.error(error)
        else:
            plan = result["plan"]
            kpis = result["kpis"]

            st.success(f"Optimization completed. Solver status: {result['status']}")

            # KPIs
            st.subheader("Key Performance Indicators")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total weekly cost (USD)", f"{kpis['total_cost']:.2f}")
            c2.metric("Avg daily cost (USD)", f"{kpis['avg_daily_cost']:.2f}")
            c3.metric("Avg daily calories (kcal)", f"{kpis['avg_daily_calories']:.0f}")

            # Weekly plan table (no sort_values to avoid key errors)
            st.subheader("Weekly meal plan")
            display_cols = [
                "day", "day_name", "meal_type",
                "Restaurant", "Meal",
                "price", "calories_kcal",
                "protein_g", "fat_g", "sugar_g",
            ]
            existing = [c for c in display_cols if c in plan.columns]
            st.dataframe(plan[existing])

            # Charts
            st.subheader("Cost per day")
            if "day_name" in plan.columns and "price" in plan.columns:
                cost_per_day = (
                    plan.groupby("day_name")["price"]
                    .sum()
                    .reindex(DAY_NAMES.values(), fill_value=0)
                )
                st.bar_chart(cost_per_day)

            st.subheader("Calories per day")
            if "day_name" in plan.columns and "calories_kcal" in plan.columns:
                cal_per_day = (
                    plan.groupby("day_name")["calories_kcal"]
                    .sum()
                    .reindex(DAY_NAMES.values(), fill_value=0)
                )
                st.bar_chart(cal_per_day)

    else:
        st.info(
            "Upload a CSV file (optional), set your preferences and click "
            "**Run optimization** to generate a weekly meal plan."
        )


if __name__ == "__main__":
    main()
