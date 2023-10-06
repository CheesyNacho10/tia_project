import random, math



class GeneratorFactory:
    IS_RENAWABLE_PROBABILITY = 0.3

    RENEWABLE_COST_PER_KW_RANGE = (0.1, 0.5)
    RENEWABLE_MAX_GENERATION_RANGE = (50, 100)
    RENEWABLE_OPERATING_COST_RANGE = (10.0, 30.0)

    NON_RENEWABLE_COST_PER_KW_RANGE = (0.5, 1.0)
    NON_RENEWABLE_MAX_GENERATION_RANGE = (100, 200)
    NON_RENEWABLE_OPERATING_COST_RANGE = (2.5, 5.0)

    @staticmethod
    def create_random_generator(total_slots, rest_slots):
        is_renewable = random.random() < GeneratorFactory.IS_RENAWABLE_PROBABILITY
        costs_per_kw = [random.uniform(*GeneratorFactory.RENEWABLE_COST_PER_KW_RANGE if is_renewable else GeneratorFactory.NON_RENEWABLE_COST_PER_KW_RANGE) for _ in range(total_slots)]
        operating_cost = [random.uniform(*GeneratorFactory.RENEWABLE_OPERATING_COST_RANGE if is_renewable else GeneratorFactory.NON_RENEWABLE_OPERATING_COST_RANGE) for _ in range(total_slots)]
        max_generation = [random.randint(*GeneratorFactory.RENEWABLE_MAX_GENERATION_RANGE if is_renewable else GeneratorFactory.NON_RENEWABLE_MAX_GENERATION_RANGE) for _ in range(total_slots)]
        return Generator(is_renewable, rest_slots, costs_per_kw, operating_cost, max_generation)

class Generator:
    def __init__(self, is_renewable, rest_slots, costs_per_kw, operating_cost, max_generation):
        self.is_renewable = is_renewable
        self.rest_slots = rest_slots
        self.costs_per_kw = costs_per_kw
        self.operating_costs = operating_cost
        self.max_generations = max_generation

class ProblemModelFactory:
    NUM_GENERATORS = (10, 20)
    TOTAL_SLOTS = (3, 4)
    REST_SLOTS = (1, 2)
    DEMAND_RANGE = (1000, 2000)
    
    @staticmethod
    def create_random_problem_model():
        total_slots = random.randint(*ProblemModelFactory.TOTAL_SLOTS)
        rest_slots = random.randint(*ProblemModelFactory.REST_SLOTS)
        generators = [GeneratorFactory.create_random_generator(total_slots, rest_slots) for _ in range(random.randint(*ProblemModelFactory.NUM_GENERATORS))]
        demands = [random.randint(*ProblemModelFactory.DEMAND_RANGE) for _ in total_slots]
        return ProblemModel(generators, demands)

class ProblemModel:
    def __init__(self, generators, demands):
        self.generators = generators
        self.demands = demands

    def respects_rest_time_slots(self, generator, generator_percentages):
        return sum([1 for percentage in generator_percentages if percentage == 0]) >= generator.rest_slots

    def fix_rest_time_slots(self, generator_percentages):
        rest_slots = self.generators[0].rest_slots
        # generator_percentages is an array of slots with the percentage of generation
        active_slots = [i for i, percentage in enumerate(generator_percentages) if percentage > 0]
        if len(active_slots) <= rest_slots:
            return generator_percentages
        
        slots_to_turn_off = len(active_slots) - rest_slots
        for _ in range(slots_to_turn_off):
            slot_to_turn_off = random.choice(active_slots)
            generator_percentages[slot_to_turn_off] = 0
            active_slots.remove(slot_to_turn_off)
        return generator_percentages

    def fix_all_rest_time_slots(self, generation_percentages):
        return [self.fix_rest_time_slots(generator_percentages) for generator_percentages in generation_percentages]

    def get_excess_energy(self, slot, generation_percentages):
        total_generation = sum([generator.get_slot_generation(slot, generation_percentages) for generator in self.generators])
        return total_generation - self.demands[slot]
    
    def get_cost_slot(self, slot, generation_percentages):
        excess_energy = self.get_excess_energy(slot)
        if excess_energy == 0:
            return 0
        
        operating_cost = sum([generator.operating_costs[slot] * math.floor(generation_percentages[generator_index][slot]) for generator, generator_index in enumerate(self.generators)])
        if excess_energy < 0: # Operation costs + the cost of the most expensive generator as excess
            return operating_cost + sorted(self.generators, key=lambda gen: -gen.costs_per_kw[slot])[0].costs_per_kw[slot] * -excess_energy
        else:
            covered_demand = self.demands[slot]
            accumulated_cost = operating_cost
            for generator, generator_index in enumerate(sorted(self.generators, key=lambda gen: gen.costs_per_kw[slot])):
                if covered_demand == 0:
                    accumulated_cost += generator.costs_per_kw[slot] * generator.max_generations[slot] * generation_percentages[generator_index][slot]
                else:
                    actual_generation = generator.max_generations[slot] * generation_percentages[generator_index][slot]
                    if actual_generation > covered_demand:
                        excess_generation = actual_generation - covered_demand
                        accumulated_cost += generator.costs_per_kw[slot] * excess_generation
                        covered_demand = 0
                    else:
                        covered_demand -= actual_generation
            return accumulated_cost

    def calculate_full_cost(self, generation_percentages):
        return sum([self.get_cost_slot(slot, generation_percentages) for slot in range(len(generation_percentages))])

    def calculate_renewable_generation(self, generation_percentages):
        return sum([sum([generator.get_slot_generation(slot, generation_percentages) for generator in self.generators if generator.is_renewable]) for slot in range(len(generation_percentages))])

class IndividualFactory:
    @staticmethod
    def create_random_individual(problem_model):
        generation_percentages = [[random.uniform(0.0, 1.0) for _ in range(len(generator.max_generations))] for generator in problem_model.generators]
        return Individual(problem_model, generation_percentages, 0)

class Individual:
    def __init__(self, problem_model, generation_percentages, generation):
        self.problem_model = problem_model
        self.generation_percentages = self.problem_model.fix_rest_time_slots(generation_percentages)
        self.generation = generation

        self.fitness = self.calculate_fitness()

    def calculate_fitness(self):
        return self.problem_model.calculate_full_cost(self.generation_percentages), self.problem_model.calculate_renewable_generation(self.generation_percentages)

    def mutate(self):
        slot_to_mutate = random.randint(0, len(self.generation_percentages) - 1)
        generator_to_mutate = random.randint(0, len(self.generation_percentages) - 1)
        self.generation_percentages[generator_to_mutate][slot_to_mutate] = random.uniform(0.1, 1.0)
        self.problem_model.fix_rest_time_slots(self.generation_percentages)
        self.fitness = self.calculate_fitness()

    def crossover(self, other):
        generator_to_crossover = random.randint(0, len(self.generation_percentages) - 1)
        firts_slot, second_slot = random.sample(range(len(self.generation_percentages[generator_to_crossover])), 2)
        firts_generation_percentages = self.generation_percentages.copy()
        second_generation_percentages = other.generation_percentages.copy()
        firts_generation_percentages[generator_to_crossover] = firts_generation_percentages[generator_to_crossover][:firts_slot] + second_generation_percentages[generator_to_crossover][firts_slot:second_slot] + firts_generation_percentages[generator_to_crossover][second_slot:]
        second_generation_percentages[generator_to_crossover] = second_generation_percentages[generator_to_crossover][:firts_slot] + firts_generation_percentages[generator_to_crossover][firts_slot:second_slot] + second_generation_percentages[generator_to_crossover][second_slot:]
        return Individual(self.problem_model, firts_generation_percentages), Individual(self.problem_model, second_generation_percentages)
    
class Poblation:
    def __init__(self, problem_model, population_size):
        self.individuals = [IndividualFactory.create_random_individual(problem_model) for _ in range(population_size)]
        self.best_individual = self.get_best_individual()
        
    def get_best_individual(self):
        return min(self.individuals, key=lambda individual: individual.fitness)