import streamlit as st
import csv
import random
import pandas as pd

#################################################################
# 1. DATA LOADING FUNCTION
#################################################################

def read_csv_to_dict(file_path):
    """Reads the CSV file and returns a dictionary of program ratings."""
    program_ratings = {}
    
    try:
        with open(file_path, mode='r', newline='', encoding='utf-8-sig') as file:
            reader = csv.reader(file)
            # Skip the header
            header = next(reader)
            
            for row in reader:
                if not row:  # Skip empty rows
                    continue
                program = row[0]
                try:
                    # Convert all rating values to floats
                    ratings = [float(x) for x in row[1:]]
                    program_ratings[program] = ratings
                except ValueError as e:
                    st.error(f"Error processing row for '{program}': {e}. Check for non-numeric ratings.")
                except IndexError:
                    st.error(f"Error processing row for '{program}': Row is shorter than expected.")

        if not program_ratings:
            st.error("No data loaded. The CSV file might be empty or formatted incorrectly.")
            
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found.")
        return {}
    except Exception as e:
        st.error(f"An unexpected error occurred while reading the file: {e}")
        return {}
        
    return program_ratings

#################################################################
# 2. GENETIC ALGORITHM CORE FUNCTIONS
#
# Note: These are refactored from your code to solve the
# 18-time-slot problem correctly.
#
#################################################################

def fitness_function(schedule, ratings_dict):
    """Calculates the total rating for a given schedule."""
    total_rating = 0
    for time_slot, program in enumerate(schedule):
        if program in ratings_dict:
            # The schedule is 18 items long, and the ratings list for each
            # program is also 18 items long (for Hour 6 to Hour 23).
            # time_slot will be 0, 1, 2... 17
            if time_slot < len(ratings_dict[program]):
                total_rating += ratings_dict[program][time_slot]
            else:
                # This should not happen if data is clean
                st.warning(f"Warning: Not enough rating data for program '{program}' at timeslot {time_slot}.")
        else:
             st.warning(f"Warning: Program '{program}' in schedule not found in ratings dictionary.")
    return total_rating

def crossover(parent1, parent2):
    """Performs single-point crossover."""
    if len(parent1) < 2: # Cannot crossover a schedule with 1 or 0 items
        return parent1, parent2
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutate(schedule, all_programs):
    """Mutates one point in the schedule."""
    if not schedule or not all_programs:
        return schedule
    
    mutation_point = random.randint(0, len(schedule) - 1)
    new_program = random.choice(all_programs)
    
    # Create a copy to avoid modifying the original schedule in place
    schedule_copy = schedule[:]
    schedule_copy[mutation_point] = new_program
    return schedule_copy

def genetic_algorithm(all_programs, num_time_slots, ratings_dict, pop_size, generations, crossover_rate, mutation_rate, elitism_size):
    """
    Runs the Genetic Algorithm to find the optimal schedule.
    
    A "schedule" (or "chromosome") is a list of 18 programs,
    one for each time slot from 6:00 to 23:00.
    """
    
    # --- 1. Initialize Population ---
    # Create a population of 'pop_size' random schedules
    population = []
    for _ in range(pop_size):
        random_schedule = [random.choice(all_programs) for _ in range(num_time_slots)]
        population.append(random_schedule)

    # --- 2. Evolution Loop ---
    for generation in range(generations):
        # Evaluate fitness of the entire population
        pop_with_fitness = []
        for schedule in population:
            fitness = fitness_function(schedule, ratings_dict)
            pop_with_fitness.append((schedule, fitness))

        # Sort by fitness (highest first)
        pop_with_fitness.sort(key=lambda x: x[1], reverse=True)

        new_population = []

        # --- 3. Elitism ---
        # Keep the best 'elitism_size' schedules
        elite_schedules = [schedule for schedule, fitness in pop_with_fitness[:elitism_size]]
        new_population.extend(elite_schedules)

        # --- 4. Crossover & Mutation (Fill the rest) ---
        while len(new_population) < pop_size:
            # Select two parents (using random selection from the whole pop for simplicity)
            parent1, parent2 = random.choices(population, k=2)

            # Crossover
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Mutation
            if random.random() < mutation_rate:
                child1 = mutate(child1, all_programs)
            if random.random() < mutation_rate:
                child2 = mutate(child2, all_programs)

            new_population.extend([child1, child2])

        # The new generation is ready
        population = new_population[:pop_size] # Trim to exact pop_size

    # --- 5. Return Best Result ---
    # After all generations, find the best schedule in the final population
    final_pop_with_fitness = []
    for schedule in population:
        fitness = fitness_function(schedule, ratings_dict)
        final_pop_with_fitness.append((schedule, fitness))
    
    final_pop_with_fitness.sort(key=lambda x: x[1], reverse=True)
    
    best_schedule = final_pop_with_fitness[0][0]
    best_fitness = final_pop_with_fitness[0][1]
    
    return best_schedule, best_fitness

#################################################################
# 3. STREAMLIT UI
#################################################################

st.set_page_config(layout="wide")
st.title("📺 TV Schedule Genetic Algorithm")

# --- Constants ---
GEN = 100 # Generations
POP = 50 # Population size
EL_S = 2 # Elitism size
# --- IMPORTANT: Change this line to match your file name ---
CSV_FILE_PATH = 'program.rating_csv' 

# --- Sidebar for User Input (Step 3) ---
st.sidebar.header("GA Parameters")
st.sidebar.markdown("""
**Step 3:** Set your GA parameters for the trial.
""")

# Crossover Rate (CO_R)
co_rate = st.sidebar.slider(
    "Crossover Rate (CO_R)",
    min_value=0.0,
    max_value=0.95,
    value=0.8, # Default from instructions
    step=0.05
)

# Mutation Rate (MUT_R)
mut_rate = st.sidebar.slider(
    "Mutation Rate (MUT_R)",
    min_value=0.01,
    max_value=0.05,
    value=0.02, # Default set within the required range
    step=0.01
)

st.sidebar.warning("""
**Note on Mutation Rate:** Your instructions specified a default of `0.2` but also a *required range* of `0.01 - 0.05`. 
I have followed the **required range** for the slider.
""")

st.sidebar.markdown("""
---
**Step 5:** Run 3 trials with different parameters.
""")

run_button = st.sidebar.button("Run New Trial")

# --- Initialize Session State to store trial results ---
if "trials" not in st.session_state:
    st.session_state.trials = []

# --- Main Logic ---
if run_button:
    with st.spinner("Running Genetic Algorithm... (This may take a moment)"):
        # 1. Load Data
        program_ratings_dict = read_csv_to_dict(CSV_FILE_PATH)
        
        if program_ratings_dict:
            all_programs = list(program_ratings_dict.keys())
            
            # The 18 time slots are 6:00 to 23:00
            # The indices 0-17 map to these 18 slots
            all_time_slots = list(range(6, 24))
            num_time_slots = len(all_time_slots)

            # 2. Run GA
            best_schedule, best_fitness = genetic_algorithm(
                all_programs=all_programs,
                num_time_slots=num_time_slots,
                ratings_dict=program_ratings_dict,
                pop_size=POP,
                generations=GEN,
                crossover_rate=co_rate,
                mutation_rate=mut_rate,
                elitism_size=EL_S
            )

            # 3. Format results for display (Step 4)
            schedule_df = pd.DataFrame({
                'Time Slot': [f"{t}:00" for t in all_time_slots],
                'Scheduled Program': best_schedule
            })

            # 4. Store trial result in session state
            trial_result = {
                "id": len(st.session_state.trials) + 1,
                "co_rate": co_rate,
                "mut_rate": mut_rate,
                "fitness": best_fitness,
                "schedule_df": schedule_df
            }
            # Add new trial to the beginning of the list
            st.session_state.trials.insert(0, trial_result)

# --- Display Trial Results (Steps 4 & 5) ---
st.header("Trial Results")

if not st.session_state.trials:
    st.info("Click 'Run New Trial' in the sidebar to start.")
else:
    st.markdown(f"You have run **{len(st.session_state.trials)}** trial(s).")
    
    # Display all completed trials
    for trial in st.session_state.trials:
        st.subheader(f"Trial {trial['id']}")
        st.markdown(f"**Parameters:** Crossover Rate = `{trial['co_rate']}`, Mutation Rate = `{trial['mut_rate']}`")
        st.markdown(f"**Resulting Total Rating (Fitness):** {trial['fitness']:.4f}")
        
        st.markdown("**Resulting Schedule:**")
        st.dataframe(trial['schedule_df'])
        st.markdown("---")
