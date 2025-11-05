import streamlit as st
import csv
import random
import pandas as pd

# ===================== STEP 1: READ CSV =====================
def read_csv_to_dict(file_path):
    program_ratings = {}
    try:
        with open(file_path, mode='r', newline='', encoding='utf-8-sig') as file:
            reader = csv.reader(file)
            header = next(reader, None)
            if not header:
                st.error("❌ CSV file is empty or has no header.")
                return {}

            for row in reader:
                if not row:
                    continue
                program = row[0].strip()
                try:
                    ratings = [float(x) if x.strip() else 0.0 for x in row[1:]]
                except ValueError:
                    ratings = [0.0 for _ in row[1:]]
                program_ratings[program] = ratings

        if not program_ratings:
            st.warning("⚠️ No valid data found in CSV.")
    except FileNotFoundError:
        st.error(f"❌ File '{file_path}' not found. Please make sure it’s in the same folder as this app.")
    return program_ratings


# ===================== STEP 2: GENETIC ALGORITHM FUNCTIONS =====================
def fitness_function(schedule, ratings):
    total_rating = 0
    for time_slot, program in enumerate(schedule):
        if program in ratings and time_slot < len(ratings[program]):
            total_rating += ratings[program][time_slot]
    return total_rating


def initialize_population(programs, pop_size):
    population = []
    for _ in range(pop_size):
        schedule = programs.copy()
        random.shuffle(schedule)
        population.append(schedule)
    return population


def crossover(schedule1, schedule2):
    point = random.randint(1, len(schedule1) - 2)
    child1 = schedule1[:point] + schedule2[point:]
    child2 = schedule2[:point] + schedule1[point:]
    return child1, child2


def mutate(schedule, all_programs):
    idx = random.randint(0, len(schedule) - 1)
    new_program = random.choice(all_programs)
    schedule[idx] = new_program
    return schedule


def genetic_algorithm(ratings, all_programs, generations, pop_size, crossover_rate, mutation_rate, elitism):
    population = initialize_population(all_programs, pop_size)
    best_fitness_history = []

    for generation in range(generations):
        population.sort(key=lambda s: fitness_function(s, ratings), reverse=True)
        new_pop = population[:elitism]  # elitism: best individuals survive

        while len(new_pop) < pop_size:
            parent1, parent2 = random.choices(population, k=2)

            # Apply crossover based on probability
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Apply mutation based on probability
            if random.random() < mutation_rate:
                mutate(child1, all_programs)
            if random.random() < mutation_rate:
                mutate(child2, all_programs)

            new_pop.extend([child1, child2])

        population = new_pop[:pop_size]
        best_fitness_history.append(fitness_function(population[0], ratings))

    best = max(population, key=lambda s: fitness_function(s, ratings))
    return best, best_fitness_history


# ===================== STREAMLIT INTERFACE =====================
st.title("🎯 Genetic Algorithm — TV Schedule Optimizer (23 Hours)")

st.sidebar.header("⚙️ Genetic Algorithm Settings")

file_path = "program_ratings_modified.csv"
ratings = read_csv_to_dict(file_path)

if ratings:
    all_programs = list(ratings.keys())
    num_slots = 23  # fixed 23 hours
    all_time_slots = [(6 + i) % 24 for i in range(num_slots)]
    time_labels = [f"{hour:02d}:00" for hour in all_time_slots]

    st.write(f"✅ Loaded {len(ratings)} programs, optimizing across {num_slots} hours (6 AM – 4 AM).")

    # GA parameters
    crossover_rate = st.sidebar.slider("Crossover Rate (e.g. 0.90)", 0.0, 1.0, 0.90, step=0.01)
    mutation_rate = st.sidebar.slider("Mutation Rate (e.g. 0.10)", 0.0, 1.0, 0.10, step=0.01)
    generations = st.sidebar.number_input("Generations", 10, 500, 100)
    population_size = st.sidebar.number_input("Population Size", 10, 500, 100)
    elitism = st.sidebar.number_input("Elitism (Top Survivors)", 1, 10, 2)

    st.info(
        """
        🔹 **Crossover (0.90)** → Combines 90% of parents’ genes to create new schedules.  
        🔹 **Mutation (0.10)** → Randomly changes 10% of the schedule to explore new possibilities.  
        🔹 **Goal:** Maximize total viewer ratings for a 23-hour schedule.
        """
    )

    # Show sample programs
    st.write("### Sample Program Ratings")
    st.dataframe(pd.DataFrame(list(ratings.items()), columns=["Program", "Ratings"]).head(5))

    if st.button("🚀 Run Genetic Algorithm"):
        with st.spinner("Optimizing TV schedule..."):
            best_schedule, fitness_history = genetic_algorithm(
                ratings, all_programs, generations, population_size, crossover_rate, mutation_rate, elitism
            )

        total_fitness = fitness_function(best_schedule, ratings)
        st.success(f"✅ Optimization Complete! Total Fitness: {round(total_fitness, 2)}")

        # Display best schedule
        st.subheader("🕒 Optimized 23-Hour TV Schedule")
        schedule_data = []
        for i, program in enumerate(best_schedule[:num_slots]):
            rating_value = (
                round(ratings[program][i], 2)
                if program in ratings and i < len(ratings[program])
                else "-"
            )
            schedule_data.append({
                "Time Slot": time_labels[i],
                "Program": program,
                "Rating": rating_value
            })
        st.dataframe(pd.DataFrame(schedule_data))

        # Plot fitness progress
        st.line_chart(fitness_history, height=200)
