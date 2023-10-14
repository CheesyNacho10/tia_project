import random, math
from typing import Annotated

class GeneratorFactory:
    IS_RENAWABLE_PROBABILITY: Annotated[float, 'The probability that the generator could be renewable'] = 0.3
    INITIAL_GENERATION_PERCENTAGE_RANGE: Annotated[tuple[float, float], 'The range of the initial generation percentage of the generator'] = (0.0, 1.0)

    RENEWABLE_COST_PER_KW_RANGE: Annotated[tuple[float, float], 'The range of the cost per kw for the renewable generators'] = (0.1, 0.5)
    RENEWABLE_MAX_GENERATION_RANGE: Annotated[tuple[int, int], 'The range of the maximum generation in kw for the renewable generators'] = (50, 100)
    RENEWABLE_OPERATING_COST_RANGE: Annotated[tuple[float, float], 'The range of the operating cost for the renewable generators'] = (10.0, 30.0)

    NON_RENEWABLE_COST_PER_KW_RANGE: Annotated[tuple[float, float], 'The range of the cost per kw for the non renewable generators'] = (0.5, 1.0)
    NON_RENEWABLE_MAX_GENERATION_RANGE: Annotated[tuple[int, int], 'The range of the maximum generation in kw for the non renewable generators'] = (100, 200)
    NON_RENEWABLE_OPERATING_COST_RANGE: Annotated[tuple[float, float], 'The range of operating cost for the non renewable generators'] = (2.5, 5.0)

    @staticmethod
    def create_random_generator(total_slots: int, rest_slots: int):
        is_renewable = random.random() < GeneratorFactory.IS_RENAWABLE_PROBABILITY
        costs_per_kw = [random.uniform(*GeneratorFactory.RENEWABLE_COST_PER_KW_RANGE if is_renewable else GeneratorFactory.NON_RENEWABLE_COST_PER_KW_RANGE) for _ in range(total_slots)]
        operating_cost = [random.uniform(*GeneratorFactory.RENEWABLE_OPERATING_COST_RANGE if is_renewable else GeneratorFactory.NON_RENEWABLE_OPERATING_COST_RANGE) for _ in range(total_slots)]
        max_generation = [random.randint(*GeneratorFactory.RENEWABLE_MAX_GENERATION_RANGE if is_renewable else GeneratorFactory.NON_RENEWABLE_MAX_GENERATION_RANGE) for _ in range(total_slots)]
        generated_percentage = [random.uniform(*GeneratorFactory.INITIAL_GENERATION_PERCENTAGE_RANGE) for _ in range(total_slots)]
        return Generator(is_renewable, rest_slots, costs_per_kw, operating_cost, max_generation, generated_percentage)

class Generator:
    is_renewable: Annotated[bool, 'Indicates if the generator is usign renewable energy']
    rest_slots: Annotated[int, 'Number of time slots that the generator must be turned off']
    costs_per_kw: Annotated[list[float], 'Cost per kw of the generator in each time slot']
    operating_costs: Annotated[list[float], 'Operating cost of the generator in each time slot']
    max_generations: Annotated[list[int], 'Maximum generation of the generator in each time slot']
    generated_percentage: Annotated[list[float], 'Percentage of generation of the generator in each time slot']

    def __init__(self, is_renewable: bool, rest_slots: int, costs_per_kw: list[float], operating_cost: list[float], max_generation: list[int], generated_percentage: list[float]):
        self.is_renewable = is_renewable
        self.rest_slots = rest_slots
        self.costs_per_kw = costs_per_kw
        self.operating_costs = operating_cost
        self.max_generations = max_generation
        self.generated_percentage = generated_percentage
        self.fix_generator_percentage()

    def fix_generator_percentage(self):
        active_slots = [index for index, percentage in enumerate(self.generated_percentage) if percentage > 0]
        if len(active_slots) <= self.rest_slots:
            return
        
        num_slots_to_turn_off = len(active_slots) - self.rest_slots
        for _ in range(num_slots_to_turn_off):
            slot_to_turn_off = random.choice(active_slots)
            self.generated_percentage[slot_to_turn_off] = 0
            active_slots.remove(slot_to_turn_off)

class ProblemModelFactory:
    NUM_GENERATORS: Annotated[tuple[int, int], 'The range of generators that could have the problem'] = (10, 20)
    TOTAL_SLOTS: Annotated[tuple[int, int], 'The range of possible total slots of time that could have the problem'] = (3, 4)
    REST_SLOTS: Annotated[tuple[int, int], 'The range of possible rests slots of time that could have the problem'] = (1, 2)
    DEMAND_RANGE: Annotated[tuple[int, int], 'The range of possible demands for every slot in the problem'] = (1000, 2000)
    
    @staticmethod
    def create_random_problem_model():
        total_slots = random.randint(*ProblemModelFactory.TOTAL_SLOTS)
        rest_slots = random.randint(*ProblemModelFactory.REST_SLOTS)
        generators = [GeneratorFactory.create_random_generator(total_slots, rest_slots) for _ in range(random.randint(*ProblemModelFactory.NUM_GENERATORS))]
        demands = [random.randint(*ProblemModelFactory.DEMAND_RANGE) for _ in range(total_slots)]
        return ProblemModel(generators, demands)

class ProblemModel:
    generators: Annotated[list[Generator], 'List of generators']
    demands: Annotated[list[int], 'List of demands for every slot']

    def __init__(self, generators, demands):
        self.generators = generators
        self.demands = demands

    def fix_rest_time_slots(self, generator_percentages):
        rest_slots = self.generators[0].rest_slots
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

    def get_excess_energy(self, slot_index, generation_percentages):
        total_generation = sum([generator.max_generations[slot_index] * generation_percentages[generator_index][slot_index] for generator_index, generator in enumerate(self.generators)])
        return total_generation - self.demands[slot_index]
    
    def get_cost_slot(self, slot_index, generation_percentages):
        excess_energy = self.get_excess_energy(slot_index, generation_percentages)
        if excess_energy == 0:
            return 0
        
        operating_cost = sum([generator.operating_costs[slot_index] * math.floor(generation_percentages[generator_index][slot_index]) for generator_index, generator  in enumerate(self.generators)])
        if excess_energy < 0: # Operation costs + the cost of the most expensive generator as excess
            return operating_cost + sorted(self.generators, key=lambda gen: -gen.costs_per_kw[slot_index])[0].costs_per_kw[slot_index] * -excess_energy
        else:
            covered_demand = self.demands[slot_index]
            accumulated_cost = operating_cost
            for generator, generator_index in enumerate(sorted(self.generators, key=lambda gen: gen.costs_per_kw[slot_index])):
                if covered_demand == 0:
                    accumulated_cost += generator.costs_per_kw[slot_index] * generator.max_generations[slot_index] * generation_percentages[generator_index][slot_index]
                else:
                    actual_generation = generator.max_generations[slot_index] * generation_percentages[generator_index][slot_index]
                    if actual_generation > covered_demand:
                        excess_generation = actual_generation - covered_demand
                        accumulated_cost += generator.costs_per_kw[slot_index] * excess_generation
                        covered_demand = 0
                    else:
                        covered_demand -= actual_generation
            return accumulated_cost

    def calculate_full_cost(self, generation_percentages):
        return sum([self.get_cost_slot(slot_index, generation_percentages) for slot_index in range(len(generation_percentages[0]))])

    def calculate_renewable_generation(self, generation_percentages):
        return sum([sum([generator.get_slot_generation(slot, generation_percentages) for generator in self.generators if generator.is_renewable]) for slot in range(len(generation_percentages))])

class IndividualFactory:
    @staticmethod
    def create_random_individual(problem_model):
        generation_percentages = [[random.uniform(0.0, 1.0) for _ in range(len(generator.max_generations))] for generator in problem_model.generators]
        return Individual(problem_model, generation_percentages, 0)

class Individual:
    problem_model: Annotated[ProblemModel, 'The problem model']
    GenerationPercentages = Annotated[list[list[float]], 'The generation percentages of every generator in every time slot']
    generation_percentages: GenerationPercentages
    generation_index: Annotated[int, 'The generation index of the individual']

    def __init__(self, problem_model, generation_percentages, generation_index):
        self.problem_model = problem_model
        self.generation_percentages = [self.problem_model.fix_rest_time_slots(generation_percentage) for generation_percentage in generation_percentages]
        self.generation_index = generation_index

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
        generation = max(self.generation_index, other.generation)
        generator_to_crossover = random.randint(0, len(self.generation_percentages) - 1)
        firts_slot, second_slot = random.sample(range(len(self.generation_percentages[generator_to_crossover])), 2)
        firts_generation_percentages = self.generation_percentages.copy()
        second_generation_percentages = other.generation_percentages.copy()
        firts_generation_percentages[generator_to_crossover] = firts_generation_percentages[generator_to_crossover][:firts_slot] + second_generation_percentages[generator_to_crossover][firts_slot:second_slot] + firts_generation_percentages[generator_to_crossover][second_slot:]
        second_generation_percentages[generator_to_crossover] = second_generation_percentages[generator_to_crossover][:firts_slot] + firts_generation_percentages[generator_to_crossover][firts_slot:second_slot] + second_generation_percentages[generator_to_crossover][second_slot:]
        return Individual(self.problem_model, firts_generation_percentages, generation + 1), Individual(self.problem_model, second_generation_percentages, generation + 1)
    
class Poblation:
    def __init__(self, problem_model, population_size):
        self.population_size = population_size
        self.individuals = [IndividualFactory.create_random_individual(problem_model) for _ in range(population_size)]
        self.best_individuals = [self.get_best_individuals()]
        self.generations = 0

    def get_best_individuals(self):
        best_cost_individual = sorted(self.individuals, key=lambda ind: ind.fitness[0])[0]
        best_renewable_individual = sorted(self.individuals, key=lambda ind: ind.fitness[1])[0]
        return best_cost_individual, best_renewable_individual

    def selection(self):
        total_fitness = sum([ind.fitness[0] for ind in self.individuals])
        selected_individuals = []
        for _ in range(len(self.individuals)):
            random_number = random.uniform(0, total_fitness)
            accumulated_fitness = 0
            for ind in self.individuals:
                accumulated_fitness += ind.fitness[0]
                if accumulated_fitness >= random_number:
                    selected_individuals.append(ind)
                    break
        return selected_individuals
    
    def evolve(self):
        # Selection
        selected_individuals = self.selection()

        # Crossover
        new_individuals = []
        for _ in range(len(self.individuals) // 2):
            first_individual, second_individual = random.sample(selected_individuals, 2)
            first_individual, second_individual = first_individual.crossover(second_individual)
            new_individuals.append(first_individual)
            new_individuals.append(second_individual)
        self.individuals = self.individuals + new_individuals

        # Mutation
        for ind in self.individuals:
            if random.random() < 0.1:
                ind.mutate()

        # Remplacement
        this_generation_individuals = list(filter(lambda ind: ind.generation_index == self.generations + 1, self.individuals))
        last_generation_individuals = list(filter(lambda ind: ind.generation_index == self.generations, self.individuals))
        
        this_generation_cost_individuals = sorted(this_generation_individuals, key=lambda ind: ind.fitness[0])[:self.population_size // 4]
        this_generation_renewable_individuals = sorted(this_generation_individuals, key=lambda ind: ind.fitness[1])[:self.population_size // 4]
        
        last_generation_cost_individuals = sorted(last_generation_individuals, key=lambda ind: ind.fitness[0])[:self.population_size // 4]
        last_generation_renewable_individuals = sorted(last_generation_individuals, key=lambda ind: ind.fitness[1])[:self.population_size // 4]
        
        self.individuals = this_generation_cost_individuals + this_generation_renewable_individuals + last_generation_cost_individuals + last_generation_renewable_individuals

        self.best_individuals.append(self.get_best_individuals())
        self.generations += 1
