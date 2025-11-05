import streamlit as st
import csv
import random
import pandas as pd

st.set_page_config(page_title="Program Rating Optimizer (Lecturer’s GA Version)", layout="wide")
st.title("📺 Program Rating Optimizer (Lecturer’s GA Version)")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("📂 Upload your program_ratings.csv file", type=["csv"])

if uploaded_file is not None:

    # ---------------- READ CSV ----------------
    def read_csv_to_dict(file_bytes):
        program_ratings = {}
        lines = file_bytes.decode("utf-8-sig").splitlines()
        reader = csv.reader(lines)
        header = next(reader, None)
        if header is None:
            st.error("CSV file appears empty or missing header.")
            return {}

        for row_idx, row in enumerate(reader, start=2):
            if not row:
                continue
            program = row[0].strip()
            try:
                ratings = [float(x) if x.strip() != "" else 0.0 for x in row[1:]]
            except ValueError:
                st.warning(f"Non-numeric rating found on CSV row {row_idx}. Replacing with 0.0 for that row.")
                ratings = []
                for x in row[1:]:
                    try:
                        ratings.append(float(x) if x.strip() != "" else 0.0)
                    except ValueError:
                        ratings.append(0.0)
            if ratings:
                program_ratings[program] = ratings

        return program_ratings

    program_ratings_dict = read_csv_to_dict(uploaded_file.read())
    if not program_ratings_dict:
        st.error("No valid program ratings loaded from CSV.")
        st.stop()

    # ---------------- PARAMETERS ----------------
    ratings = program_ratings_dict
    NUM_SLOTS = 18  # 06:00 → 23:00 only
    all_time_slots = list(range(6, 24))  # from 06:00 to 23:00
    time_labels = [f"{h:02d}:00" for h in all_time_slots]

    # ✅ Fixed GA parameters
    GEN = 100
    POP = 50
    EL_S = 2

    all_programs = list(ratings.keys())
    st.write(f"✅ Loaded **{len(ratings)}** programs. Optimizing across **{NUM_SLOTS}** hourly slots (06:00 → 23:00).")
    st.write(f"⚙️ Fixed GA Settings → Generations: {GEN}, Population: {POP}, Elitism: {EL_S}")

    # ---------------- SLIDERS FOR 3 TRIALS ----------------
    st.sidebar.header("⚙️ GA Parameters for 3 Trials (adjust Crossover & Mutation only)")
    trial_params = []
    default_settings = [(0.80, 0.10), (0.70, 0.30), (0.90, 0.05)]
    for i in range(1, 3 + 1):
        st.sidebar.subheader(f"Trial {i}")
        d_co, d_mut = default_settings[i - 1]
        co_r = st.sidebar.slider(f"Trial {i} - Crossover Rate", 0.0, 1.0, float(d_co), 0.01, key=f"co_r_{i}")
        mut_r = st.sidebar.slider(f"Trial {i} - Mutation Rate", 0.0, 1.0, float(d_mut), 0.01, key=f"mut_r_{i}")
        trial_params.append((co_r, mut_r))

    st.write("### Sample Program Ratings (first 5 programs)")
    sample_df = pd.DataFrame(list(ratings.items()), columns=["Program", "Ratings"]).head(5)
    st.dataframe(sample_df)

    # ---------------- GA HELPERS ----------------
    def fitness_function(schedule):
        total_rating = 0.0
        for time_slot, program in enumerate(schedule):
            if program in ratings and time_slot < len(ratings[program]):
                total_rating += ratings[program][time_slot]
        return total_rating

    def initialize_population(program_list, population_size, num_slots):
        population = []
        for _ in range(population_size):
            schedule = random.choices(program_list, k=num_slots)
            population.append(schedule[:num_slots])
        return population

    def crossover(schedule1, schedule2):
        length = min(len(schedule1), len(schedule2), NUM_SLOTS)
        if length < 2:
            return schedule1.copy(), schedule2.copy()
        point = random.randint(1, length - 1)
        child1 = (schedule1[:point] + schedule2[point:])[:NUM_SLOTS]
        child2 = (schedule2[:point] + schedule1[point:])[:NUM_SLOTS]
        return child1, child2

    def mutate(schedule, program_list):
        if not schedule:
            return schedule
        idx = random.randrange(len(schedule))
        schedule[idx] = random.choice(program_list)
        return schedule[:NUM_SLOTS]

    def genetic_algorithm(program_list, num_slots, generations, population_size,
                          crossover_rate, mutation_rate, elitism_size):
        population = initialize_population(program_list, population_size, num_slots)
        fitness_history = []

        for gen in range(generations):
            population.sort(key=lambda s: fitness_function(s), reverse=True)
            best_fitness = fitness_function(population[0])
            fitness_history.append(best_fitness)

            new_pop = population[:elitism_size]
            while len(new_pop) < population_size:
                parent1, parent2 = random.choices(population, k=2)
                if random.random() < crossover_rate:
                    child1, child2 = crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                if random.random() < mutation_rate:
                    child1 = mutate(child1, program_list)
                if random.random() < mutation_rate:
                    child2 = mutate(child2, program_list)
                new_pop.append(child1[:num_slots])
                if len(new_pop) < population_size:
                    new_pop.append(child2[:num_slots])
            population = new_pop

        population.sort(key=lambda s: fitness_function(s), reverse=True)
        best = population[0][:num_slots]
        return best, fitness_history

    # ---------------- RUN BUTTON ----------------
    if st.button("🚀 Run 3 Trials"):
        st.subheader("🎯 Results of 3 Trials")
        trial_results = []

        for i, (co_r, mut_r) in enumerate(trial_params, start=1):
            with st.spinner(f"Running Trial {i} (Crossover={co_r}, Mutation={mut_r})..."):
                best_schedule, fitness_history = genetic_algorithm(
                    all_programs,
                    NUM_SLOTS,
                    generations=GEN,
                    population_size=POP,
                    crossover_rate=co_r,
                    mutation_rate=mut_r,
                    elitism_size=EL_S
                )
            total_rating = fitness_function(best_schedule)
            trial_results.append({
                "trial": i,
                "schedule": best_schedule,
                "fitness": total_rating,
                "crossover": co_r,
                "mutation": mut_r,
                "history": fitness_history
            })
            st.write(f"**Trial {i}** — Crossover: `{co_r}` | Mutation: `{mut_r}` | Total Rating: **{total_rating:.2f}**")
            preview_df = pd.DataFrame({
                "Time Slot": time_labels,
                "Program": best_schedule[:NUM_SLOTS],
                "Rating": [
                    round(ratings[p][idx], 2) if p in ratings and idx < len(ratings[p]) else 0.0
                    for idx, p in enumerate(best_schedule[:NUM_SLOTS])
                ]
            })
            st.dataframe(preview_df, use_container_width=True, height=680)

        best_trial = max(trial_results, key=lambda x: x["fitness"])
        st.subheader(f"🏆 Best Schedule — Trial {best_trial['trial']}")
        best_df = pd.DataFrame({
            "Time Slot": time_labels,
            "Program": best_trial["schedule"][:NUM_SLOTS],
            "Rating": [
                round(ratings[p][idx], 2) if p in ratings and idx < len(ratings[p]) else 0.0
                for idx, p in enumerate(best_trial["schedule"][:NUM_SLOTS])
            ]
        })
        st.dataframe(best_df, use_container_width=True, height=600)
        st.success(f"✅ Best Total Ratings: {best_trial['fitness']:.2f} | Crossover: {best_trial['crossover']} | Mutation: {best_trial['mutation']}")

        # show charts
        st.subheader("📈 Fitness Progress (per generation)")
        cols = st.columns(3)
        for col, tr in zip(cols, trial_results):
            col.write(f"Trial {tr['trial']} (C={tr['crossover']}, M={tr['mutation']})")
            col.line_chart(pd.DataFrame({"Fitness": tr["history"]}))

else:
    st.info("👆 Please upload a CSV file to start.")
