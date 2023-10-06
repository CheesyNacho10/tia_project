import random, math

def created_values_in_range(generation_function, value_range, slots):
    lower, upper = value_range
    return [generation_function(lower, upper) for _ in range(slots)]

def random_double_values_in_range(value_range, slots):
    return created_values_in_range(random.uniform, value_range, slots)

def random_int_values_in_range(value_range, slots):
    return created_values_in_range(random.randint, value_range, slots)

# Constants
NUM_GENERATORS = (5, 10)
CONSUMER_DEMAND_RANGE = (500, 1000)

RENEWABLE_COST_PER_KW_RANGE = (0.1, 0.5)
RENEWABLE_MAX_GENERATION_RANGE = (50, 100)
RENEWABLE_OPERATING_COST_RANGE = (10.0, 30.0)

NON_RENEWABLE_COST_PER_KW_RANGE = (0.5, 1.0)
NON_RENEWABLE_MAX_GENERATION_RANGE = (100, 200)
NON_RENEWABLE_OPERATING_COST_RANGE = (2.5, 5.0)

class Generator:
    def __init__(self, is_renewable, rest_slots, costs_per_kw, operating_cost, max_generation):
        self.is_renewable = is_renewable
        self.rest_slots = rest_slots
        self.costs_per_kw = costs_per_kw
        self.operating_costs = operating_cost
        self.max_generations = max_generation
    
    def respects_rest_time_slots(self, generation_percentages):
        return self.rest_slots >= sum([1 for percentage in generation_percentages if percentage > 0])
    
    def fix_rest_time_slots(self, generation_percentages):
        if self.respects_rest_time_slots(generation_percentages):
            return

        active_slots = [i for i, percentage in enumerate(generation_percentages) if percentage > 0]
        slots_to_turn_off = len(active_slots) - (len(generation_percentages) - self.rest_slots)
        for _ in range(slots_to_turn_off):
            slot_to_turn_off = random.choice(active_slots)
            generation_percentages[slot_to_turn_off] = 0
            active_slots.remove(slot_to_turn_off)
        return generation_percentages

    def get_slot_generation(self, slot, generation_percentages):
        return generation_percentages[slot] * self.max_generations[slot]

    def calculate_full_cost(self, generation_percentages):
        return sum([self.calculate_slot_cost(slot) for slot in range(len(generation_percentages))])

    def calculate_slot_cost(self, slot, generation_percentages):
        return self.costs_per_kw[slot] * self.max_generations[slot] * generation_percentages[slot] \
            + self.operating_costs[slot] * math.floor(generation_percentages[slot]);

class ProblemModel:
    def __init__(self, generators, demands):
        self.generators = generators
        self.demands = demands

    def get_excess_energy(self, slot, generation_percentages):
        total_generation = sum([generator.get_slot_generation(slot, generation_percentages) for generator in self.generators])
        return total_generation - self.demands[slot]
    
    def get_cost_slot(self, slot, generation_percentages):
        excess_energy = self.get_excess_energy(slot)
        if excess_energy == 0:
            return 0
        excess_energy = abs(excess_energy)

        sorted_generators = sorted(self.generators, key=lambda gen: -gen.costs_per_kw[slot])
        cost = 0
        for generator in sorted_generators:
            generation = generator.get_slot_generation(slot, generation_percentages)
            if generation >= excess_energy:
                cost += excess_energy * generator.costs_per_kw[slot]
                break
            else:
                cost += generation * generator.costs_per_kw[slot]
                excess_energy -= generation
        return cost
    
    def get_cost(self, generation_percentages):
        return sum([self.get_cost_slot(slot, generation_percentages) for slot in range(len(self.demands))])
    
    def get_renewable_generation(self, generation_percentages):
        return sum([generator.get_slot_generation(slot, generation_percentages) for slot, generator in enumerate(self.generators) if generator.is_renewable])
    
    def fix_genotypes(self, generation_percentages):
        for generator in self.generators:
            generator.fix_rest_time_slots(generation_percentages)
        return generation_percentages

def create_random_generator_percentages(generator_model):
    generator_percentages = [0.0 for _ in range(len(generator_model.max_generations))]
    fullfilling_slots = random.sample(range(len(generator_model.max_generations)), random.randint(0, generator_model.rest_slots))
    for slot in fullfilling_slots:
        generator_percentages[slot] = random.uniform(0.1, 1.0)
    return generator_percentages

def create_random_generator_model(time_slots, rest_time_slots):
    if (rest_time_slots > time_slots):
        raise Exception("rest_time_slots cannot be greater than time_slots")

    is_renewable = random.choice([True, False])
    rest_time_slots = rest_time_slots;
    
    if is_renewable:
        cost_per_kw = random_double_values_in_range(RENEWABLE_COST_PER_KW_RANGE, time_slots)
        max_generation = random_int_values_in_range(RENEWABLE_MAX_GENERATION_RANGE, time_slots)
        operating_cost =  random_double_values_in_range(RENEWABLE_OPERATING_COST_RANGE, time_slots)
    else:
        cost_per_kw = random_double_values_in_range(NON_RENEWABLE_COST_PER_KW_RANGE, time_slots)
        max_generation = random_int_values_in_range(NON_RENEWABLE_MAX_GENERATION_RANGE, time_slots)
        operating_cost = random_double_values_in_range(NON_RENEWABLE_OPERATING_COST_RANGE, time_slots)

    return Generator(is_renewable, rest_time_slots, cost_per_kw, operating_cost, max_generation)

def create_random_problem(demands, time_slots, rest_time_slots):
    generators = [create_random_generator_model(time_slots, rest_time_slots) for _ in range(random.randint(NUM_GENERATORS[0], NUM_GENERATORS[1]))]
    return ProblemModel(generators, demands)

def create_random_population(population_size, time_slots, rest_time_slots):
    demands = random_int_values_in_range(CONSUMER_DEMAND_RANGE, time_slots)
    return [create_random_problem(demands, time_slots, rest_time_slots) for _ in range(population_size)]

def get_individual_fitness(problem_model, generation_percentages):
    problem_model.fix_genotypes(generation_percentages)
    return problem_model.get_cost(generation_percentages), problem_model.get_renewable_generation(generation_percentages)

def get_random_index(probabilities, exclude_indexes):
    random_value = random.random()
    total_probability = 0
    for i, probability in enumerate(probabilities):
        if i in exclude_indexes:
            continue
        total_probability += probability
        if total_probability >= random_value:
            return i
    return len(probabilities) - 1

def get_selection_by_wheel(problem_model, population, selection_size, last_worse_fitness):
    fitnesses = [get_individual_fitness(problem_model, individual) for individual in population]
    total_cost, total_renewal = sum([fitness[0] - last_worse_fitness for fitness in fitnesses]), sum([fitness[1] - last_worse_fitness for fitness in fitnesses])
    costs_fitneses, renewals_fitnesses = [fitness[0] - last_worse_fitness for fitness in fitnesses], [fitness[1] - last_worse_fitness for fitness in fitnesses]
    costs_probabilities, renewals_probabilities = [cost / total_cost for cost in costs_fitneses], [renewal / total_renewal for renewal in renewals_fitnesses]
    selected_indexes = []
    for _ in range(selection_size):
        selected_indexes.append(population[get_random_index(costs_probabilities, selected_indexes)])
    return list(filter(lambda individual: individual not in selected_indexes, population))

def get_two_point_crossover(problem_model, firts_individual, second_individual):
    firts_point, second_point = random.sample(range(len(firts_individual)), 2)
    if firts_point > second_point:
        firts_point, second_point = second_point, firts_point
    
    firts_child = firts_individual[:firts_point] + second_individual[firts_point:second_point] + firts_individual[second_point:]
    problem_model.fix_genotypes(firts_individual)
    second_child = second_individual[:firts_point] + firts_individual[firts_point:second_point] + second_individual[second_point:]
    return firts_child, second_child

def get_exchange_mutation(individual):
    firts_point, second_point = random.sample(range(len(individual)), 2)
    individual[firts_point], individual[second_point] = individual[second_point], individual[firts_point]
    return individual


