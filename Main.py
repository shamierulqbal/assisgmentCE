import streamlit as st
import csv
import random
import pandas as pd
import gc

st.set_page_config(page_title="📺 Program Rating Optimizer", layout="wide")
st.title("📺 Program Rating Optimizer (Lecturer’s GA Version)")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("📂 Upload your program_ratings.csv file", type=["csv"])

@st.cache_data
def read_csv_to_dict(file):
    program_ratings = {}
    reader = csv.reader(file.read().decode("utf-8").splitlines())
    header = next(reader)  # skip header

    for row in reader:
        program = row[0]
        ratings = [float(x) for x in row[1:]]
        program_ratings[program] = ratings

    return program_ratings


if uploaded_file is not None:
    program_ratings_dict = read_csv_to_dict(uploaded_file)

    ratings = program_ratings_dict
    GEN = 100
    POP = 50
    EL_S = 2

    all_programs = list(ratings.keys())
    all_time_slots = list(range(6, 24))

    # ---------------- SLIDERS FOR 3 TRIALS ----------------
    st.sidebar.header("⚙️ GA Parameters for 3 Trials")

    trial_params = []
    for i in range(1, 4):
        st.sidebar.subheader(f"Trial {i}")
        co_r = st.sidebar.slider(f"Trial {i} - Crossover Rate", 0.0, 0.95, 0.8, 0.01, key=f"co_r_{i}")
        mut_r = st.sidebar.slider(f"Trial {i} - Mutation Rate", 0.01, 0.05, 0.02, 0.01, key=f"mut_r_{i}")
        trial_params.append((co_r, mut_r))

    # ---------------- FUNCTIONS ----------------
    def fitness_function(schedule):
        total_rating = 0
        for time_slot, program in enumerate(schedule):
            total_rating += ratings[program][time_slot]
        return total_rating

    # SAFE initialization (no factorial explosion)
    def initialize_pop(programs, time_slots):
        # Limit recursion for safety
        if len(programs) > 10:
            return [random.sample(programs, len(programs))]
        if not programs:
            return [[]]
        all_schedules = []
        for i in range(len(programs)):
            for schedule in initialize_pop(programs[:i] + programs[i + 1:], time_slots):
                all_schedules.append([programs[i]] + schedule)
        return all_schedules

    # Try to build initial population safely
    try:
        all_possible_schedules = initialize_pop(all_programs, all_time_slots)
        best_schedule = random.choice(all_possible_schedules)
    except RecursionError:
        # Fallback if still too large
        best_schedule = random.sample(all_programs, len(all_programs))

    def finding_best_schedule(all_schedules):
        best_schedule = []
        max_ratings = 0
        for schedule in all_schedules:
            total_ratings = fitness_function(schedule)
            if total_ratings > max_ratings:
                max_ratings = total_ratings
                best_schedule = schedule
        return best_schedule if best_schedule else random.sample(all_programs, len(all_programs))

    def crossover(schedule1, schedule2):
        crossover_point = random.randint(1, len(schedule1) - 2)
        child1 = schedule1[:crossover_point] + schedule2[crossover_point:]
        child2 = schedule2[:crossover_point] + schedule1[crossover_point:]
        return child1, child2

    def mutate(schedule):
        mutation_point = random.randint(0, len(schedule) - 1)
        new_program = random.choice(all_programs)
        schedule[mutation_point] = new_program
        return schedule

    def genetic_algorithm(initial_schedule, generations=GEN, population_size=POP,
                          crossover_rate=0.8, mutation_rate=0.02, elitism_size=EL_S):
        population = [initial_schedule]
        for _ in range(population_size - 1):
            random_schedule = initial_schedule.copy()
            random.shuffle(random_schedule)
            population.append(random_schedule)

        for generation in range(generations):
            new_population = []
            population.sort(key=lambda schedule: fitness_function(schedule), reverse=True)
            new_population.extend(population[:elitism_size])

            while len(new_population) < population_size:
                parent1, parent2 = random.choices(population, k=2)
                if random.random() < crossover_rate:
                    child1, child2 = crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                if random.random() < mutation_rate:
                    child1 = mutate(child1)
                if random.random() < mutation_rate:
                    child2 = mutate(child2)

                new_population.extend([child1, child2])

            population = new_population

        best_final = max(population, key=lambda s: fitness_function(s))
        return best_final

    # ---------------- RUN BUTTON ----------------
    if st.button("🚀 Run 3 Trials"):
        st.subheader("🎯 Results of 3 Trials")
        trial_results = []

        initial_best_schedule = best_schedule
        rem_t_slots = len(all_time_slots) - len(initial_best_schedule)

        progress = st.progress(0)
        for i, (co_r, mut_r) in enumerate(trial_params, start=1):
            genetic_schedule = genetic_algorithm(
                initial_best_schedule,
                generations=GEN,
                population_size=POP,
                crossover_rate=co_r,
                mutation_rate=mut_r,
                elitism_size=EL_S
            )
            final_schedule = initial_best_schedule + genetic_schedule[:rem_t_slots]
            total_rating = fitness_function(final_schedule)
            trial_results.append((i, final_schedule, total_rating, co_r, mut_r))

            st.write(f"**Trial {i}** — Crossover: `{co_r}` | Mutation: `{mut_r}` | Total Rating: **{total_rating:.2f}**")
            progress.progress(i / 3)

        # ---------------- DISPLAY BEST RESULT IN TABLE ----------------
        best_trial = max(trial_results, key=lambda x: x[2])
        best_num, best_schedule, best_total, best_co, best_mut = best_trial

        st.subheader(f"🏆 Best Schedule — Trial {best_num}")
        df = pd.DataFrame({
            "Time Slot": [f"{t:02d}:00" for t in all_time_slots],
            "Program": best_schedule[:len(all_time_slots)]
        })
        st.dataframe(df, use_container_width=True)
        st.success(f"✅ Best Total Ratings: {best_total:.2f} | Crossover: {best_co} | Mutation: {best_mut}")

        gc.collect()

else:
    st.info("👆 Please upload a CSV file to start.")
