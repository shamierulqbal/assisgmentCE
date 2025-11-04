import streamlit as st
import csv
import random
import pandas as pd

# ===================== STEP 1: READ CSV =====================

def read_csv_to_dict(file_path):
    program_ratings = {}

    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)  # skip header

        for row in reader:
            program = row[0]
            ratings = [float(x) if x else 0.0 for x in row[1:]]
            program_ratings[program] = ratings

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
        new_pop = population[:elitism]

        while len(new_pop) < pop_size:
            parent1, parent2 = random.choices(population, k=2)

            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

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

st.title("📺 Genetic Algorithm TV Schedule Optimizer")

st.sidebar.header("⚙️ Parameters")

# Upload CSV file
uploaded_file = st.sidebar.file_uploader("Upload Program Ratings CSV", type=["csv"])


    # GA Parameters
    GEN = st.sidebar.slider("Generations", 10, 500, 100, step=10)
    POP = st.sidebar.slider("Population Size", 10, 300, 100, step=10)
    CO_R = st.sidebar.slider("Crossover Rate", 0.0, 1.0, 0.8, step=0.05)
    MUT_R = st.sidebar.slider("Mutation Rate", 0.0, 1.0, 0.05, step=0.01)
    EL_S = st.sidebar.slider("Elitism Size", 1, 10, 2)

    st.write("### Loaded Programs (Sample)")
    sample_df = pd.DataFrame(list(ratings.items()), columns=["Program", "Ratings"]).head(5)
    st.dataframe(sample_df)

    if st.button("🚀 Run Genetic Algorithm"):
        with st.spinner("Running Genetic Algorithm..."):
            best_schedule, fitness_history = genetic_algorithm(
                ratings, all_programs, GEN, POP, CO_R, MUT_R, EL_S
            )

        # Show final schedule
        st.success("✅ Optimization Completed!")
        total_fitness = fitness_function(best_schedule, ratings)

        st.write("### 🏆 Optimal Schedule")
        schedule_data = []
        for i, program in enumerate(best_schedule):
            schedule_data.append({
                "Time Slot": f"{all_time_slots[i]:02d}:00",
                "Program": program,
                "Rating": round(ratings[program][i], 2) if i < len(ratings[program]) else "-"
            })
        st.dataframe(pd.DataFrame(schedule_data))

        st.metric(label="Total Fitness (Total Ratings)", value=round(total_fitness, 2))

        # Plot fitness over generations
        st.write("### 📈 Fitness Progress")
        st.line_chart(fitness_history)

else:
    st.info("👆 Please upload your `program_ratings_modified.csv` file to begin.")
