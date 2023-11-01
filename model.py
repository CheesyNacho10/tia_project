import random, math
from typing import Annotated, List, Tuple
from copy import deepcopy

class Generator:
    is_renewable: Annotated[bool, 'Indicates if the generator is usign renewable energy']
    rest_slots: Annotated[int, 'Number of time slots that the generator must be turned off']
    costs_per_kw: Annotated[List[float], 'Cost per kw of the generator in each time slot']
    operating_costs: Annotated[List[float], 'Operating cost of the generator in each time slot']
    max_generations: Annotated[List[int], 'Maximum generation of the generator in each time slot']
    generated_percentage: Annotated[List[float], 'Percentage of generation of the generator in each time slot']

    def __init__(self, is_renewable: bool, rest_slots: int, costs_per_kw: List[float], 
                 operating_cost: List[float], max_generation: List[int], generated_percentage: List[float]):
        self.is_renewable = is_renewable
        self.total_slots = len(costs_per_kw)
        self.rest_slots = rest_slots
        self.costs_per_kw = costs_per_kw
        self.operating_costs = operating_cost
        self.max_generations = max_generation
        self.generated_percentage = generated_percentage
        self.fix_generator_percentage()

    def print(self, identation = ''):
        # Print its caracteristics
        print(f'{identation}Características del Generador:')
        print(f'{identation}Es Renovable: {self.is_renewable}')
        print(f'{identation}Slots de Descanso: {self.rest_slots}')
        print(f'{identation}Costo por KW: {self.costs_per_kw}')
        print(f'{identation}Costo de Operación: {self.operating_costs}')
        print(f'{identation}Máxima Generación: {self.max_generations}')
        print(f'{identation}Porcentaje de Generación: {self.generated_percentage}')
        print('')

    def fix_generator_percentage(self):
        active_slots = [index for index, percentage in enumerate(self.generated_percentage) if percentage > 0]
        if len(active_slots) <= self.rest_slots:
            return
        
        num_slots_to_turn_off = len(active_slots) - self.rest_slots
        for _ in range(num_slots_to_turn_off):
            slot_to_turn_off = random.choice(active_slots)
            self.generated_percentage[slot_to_turn_off] = 0
            active_slots.remove(slot_to_turn_off)

    def get_slot_generation(self, slot_index):
        return self.max_generations[slot_index] * self.generated_percentage[slot_index]
    
    def get_partial_cost_slot(self, slot_index, wasted_energy):
        return self.costs_per_kw[slot_index] * wasted_energy + self.operating_costs[slot_index] * math.ceil(self.generated_percentage[slot_index])
    
    def get_full_cost_slot(self, slot_index):
        return self.get_partial_cost_slot(slot_index, self.max_generations[slot_index] * self.generated_percentage[slot_index])

    def mutate(self):
        slot_to_mutate = random.randint(0, len(self.generated_percentage) - 1)

        if random.random() < 0.5:
            new_percentage = 0.0
        else:
            new_percentage = random.uniform(0.1, 1.0)
            while new_percentage == self.generated_percentage[slot_to_mutate]:
                new_percentage = random.uniform(0.1, 1.0)

        self.generated_percentage[slot_to_mutate] = new_percentage
        self.fix_generator_percentage()

    @staticmethod
    def crossover(first_generator, second_generator):
        first_slot, second_slot = random.sample(range(len(first_generator.generated_percentage)), 2)
        firts_generation_percentages = deepcopy(first_generator.generated_percentage)
        second_generation_percentages = deepcopy(second_generator.generated_percentage)
        
        # One point crossover
        firts_generation_percentages[first_slot], second_generation_percentages[second_slot] = \
            second_generation_percentages[second_slot], firts_generation_percentages[first_slot]

        # Two point crossover
        # firts_generation_percentages = firts_generation_percentages[:first_slot] + second_generation_percentages[first_slot:second_slot] + firts_generation_percentages[second_slot:]
        # second_generation_percentages = second_generation_percentages[:first_slot] + firts_generation_percentages[first_slot:second_slot] + second_generation_percentages[second_slot:]

        return Generator(first_generator.is_renewable, first_generator.rest_slots, first_generator.costs_per_kw, first_generator.operating_costs, first_generator.max_generations, firts_generation_percentages), \
            Generator(second_generator.is_renewable, second_generator.rest_slots, second_generator.costs_per_kw, second_generator.operating_costs, second_generator.max_generations, second_generation_percentages)

class GeneratorFactory:
    INITIAL_GENERATION_PERCENTAGE_RANGE: Annotated[Tuple[float, float], 'The range of the initial generation percentage of the generator'] \
        = (0.0, 1.0)

    RENEWABLE_COST_PER_KW_RANGE: Annotated[Tuple[float, float], 'The range of the cost per kw for the renewable generators'] \
        = (0.1, 0.5)
    RENEWABLE_MAX_GENERATION_RANGE: Annotated[Tuple[int, int], 'The range of the maximum generation in kw for the renewable generators'] \
        = (25, 50)
    RENEWABLE_OPERATING_COST_RANGE: Annotated[Tuple[float, float], 'The range of the operating cost for the renewable generators'] \
        = (10.0, 30.0)

    NON_RENEWABLE_COST_PER_KW_RANGE: Annotated[Tuple[float, float], 'The range of the cost per kw for the non renewable generators'] \
        = (0.5, 1.0)
    NON_RENEWABLE_MAX_GENERATION_RANGE: Annotated[Tuple[int, int], 'The range of the maximum generation in kw for the non renewable generators'] \
        = (50, 75)
    NON_RENEWABLE_OPERATING_COST_RANGE: Annotated[Tuple[float, float], 'The range of operating cost for the non renewable generators'] \
        = (2.5, 5.0)

    @staticmethod
    def create_random_generator(total_slots: int, rest_slots: int):
        is_renewable = None
        costs_per_kw = [random.uniform(*GeneratorFactory.RENEWABLE_COST_PER_KW_RANGE if is_renewable else GeneratorFactory.NON_RENEWABLE_COST_PER_KW_RANGE) \
                        for _ in range(total_slots)]
        operating_cost = [random.uniform(*GeneratorFactory.RENEWABLE_OPERATING_COST_RANGE if is_renewable else GeneratorFactory.NON_RENEWABLE_OPERATING_COST_RANGE) \
                          for _ in range(total_slots)]
        max_generation = [random.randint(*GeneratorFactory.RENEWABLE_MAX_GENERATION_RANGE if is_renewable else GeneratorFactory.NON_RENEWABLE_MAX_GENERATION_RANGE) \
                          for _ in range(total_slots)]
        generated_percentage = [random.uniform(*GeneratorFactory.INITIAL_GENERATION_PERCENTAGE_RANGE) \
                                for _ in range(total_slots)]
        return Generator(is_renewable, rest_slots, costs_per_kw, \
                         operating_cost, max_generation, generated_percentage)
    
    @staticmethod
    def create_generator(generator: Generator, is_renewable: bool):
        generated_percentage = [random.uniform(*GeneratorFactory.INITIAL_GENERATION_PERCENTAGE_RANGE) \
                                for _ in range(generator.total_slots)]
        return Generator(is_renewable, generator.rest_slots, generator.costs_per_kw, \
                            generator.operating_costs, generator.max_generations, generated_percentage)

class Individual:
    generators: Annotated[List[Generator], 'List of generators']
    demands: Annotated[List[int], 'List of demands for every slot']
    generation_index: Annotated[int, 'The index of the generation of the individual']
    fitness: Annotated[Tuple[float, float], 'The fitness of the individual. The first value is the cost and the second value is the renewable generation']

    def __init__(self, generators, demands, generation_index = 0):
        self.generators = generators
        for generator in self.generators:
            generator.generated_percentage = [random.uniform(*GeneratorFactory.INITIAL_GENERATION_PERCENTAGE_RANGE) \
                                                for _ in range(len(demands))]
            generator.fix_generator_percentage()
        self.demands = demands
        self.generation_index = generation_index
        self.fitness = (None, None)
        self.calculate_fitness()

    def print(self, identation = ''):
        print(f'{identation}Características del Individuo:')
        print(f'{identation}Fitness: {self.fitness[0]} (Costo), {self.fitness[1]} (Generación Renovable)')
        print(f'{identation}Cantidad de Generadores: {len(self.generators)}')
        print(f'{identation}Cantidad de Slots: {len(self.demands)}')
        print(f'{identation}Generadores:')
        for generator in self.generators:
            generator.print(identation + '  ')
        print('')

    def get_cost_slot(self, slot_index):
        slot_cost = 0
        slot_demand = self.demands[slot_index]
        shorted_generators = sorted(self.generators, key=lambda generator: generator.costs_per_kw[slot_index])
        
        for generator in shorted_generators:
            generation = generator.get_slot_generation(slot_index)
            if generation == 0:
                break

            if slot_demand > generation:
                slot_demand -= generation
            elif slot_demand <= generation:
                slot_cost += generator.get_partial_cost_slot(slot_index, generation - slot_demand)
                slot_demand = 0
            elif slot_demand == 0:
                slot_cost += generator.get_full_cost_slot(slot_index)
        
        if slot_demand > 0:
            slot_cost += shorted_generators[-1].get_partial_cost_slot(slot_index, slot_demand)

        return slot_cost

    def calculate_full_cost(self):
        return sum([self.get_cost_slot(slot) for slot in range(len(self.demands))])

    def calculate_renewable_generation(self):
        return sum([sum([generator.get_slot_generation(slot) for generator in self.generators if generator.is_renewable]) for slot in range(len(self.demands))])

    def calculate_fitness(self):
        self.fitness = self.calculate_full_cost(), self.calculate_renewable_generation()

    def mutate(self):
        generator_to_mutate_index = random.randint(0, len(self.generators) - 1)
        self.generators[generator_to_mutate_index].mutate()
        self.calculate_fitness()

    def get_neighbor(self):
        neighbor = deepcopy(self)
        neighbor.mutate()
        return neighbor

    @staticmethod
    def crossover(firts_individual, second_individual):
        max_generation_index = max(firts_individual.generation_index, second_individual.generation_index)
        firts_individual, second_individual = deepcopy(firts_individual), deepcopy(second_individual)
        firts_generator_index, second_generator_index = random.sample(range(len(firts_individual.generators)), 2)
        firts_individual.generators[firts_generator_index], second_individual.generators[second_generator_index] = \
            Generator.crossover(firts_individual.generators[firts_generator_index], second_individual.generators[second_generator_index])
        return Individual(firts_individual.generators, firts_individual.demands, max_generation_index + 1), \
            Individual(second_individual.generators, second_individual.demands, max_generation_index + 1)

class PopulationFactory:
    IS_RENEWABLE_PERCENTAGE_RANGE: Annotated[float, 'The probability that the generator could be renewable'] \
        = (0.3, 0.5)
    NUM_GENERATORS: Annotated[Tuple[int, int], 'The range of generators that could have the problem'] \
        = (10, 20)
    TOTAL_SLOTS: Annotated[Tuple[int, int], 'The range of possible total slots of time that could have the problem'] \
        = (3, 4)
    REST_SLOTS: Annotated[Tuple[int, int], 'The range of possible rests slots of time that could have the problem'] \
        = (1, 2)
    DEMAND_RANGE: Annotated[Tuple[int, int], 'The range of possible demands for every slot in the problem'] \
        = (500, 700)
    
    @staticmethod
    def create_random_population(population_size):
        renwable_percentage = random.uniform(*PopulationFactory.IS_RENEWABLE_PERCENTAGE_RANGE)
        num_generators = random.randint(*PopulationFactory.NUM_GENERATORS)
        total_slots = random.randint(*PopulationFactory.TOTAL_SLOTS)
        rest_slots = random.randint(*PopulationFactory.REST_SLOTS)

        renewable_generators = int(num_generators * renwable_percentage)
        non_renewable_generators = num_generators - renewable_generators

        generator_model = GeneratorFactory.create_random_generator(total_slots, rest_slots)
        generators = [GeneratorFactory.create_generator(generator_model, False) for _ in range(non_renewable_generators)]
        generators.extend([GeneratorFactory.create_generator(generator_model, True) for _ in range(renewable_generators)])
        demands = [random.randint(*PopulationFactory.DEMAND_RANGE) for _ in range(total_slots)]
        return Population([Individual(generators, demands) for _ in range(population_size)])

class IndividualFactory:

    @staticmethod
    def create_random_individual():
        return PopulationFactory.create_random_population(1).individuals[0]

class Population:
    SELECTION_FACTOR: Annotated[float, 'The factor of the population that could be selected'] \
        = 1.0
    CROSSOVER_FACTOR: Annotated[float, 'The factor of the population that could be crossed'] \
        = 0.5
    MAX_INDIVIDUALS: Annotated[int, 'The maximum number of individuals that the population could have'] \
        = 1000
    MAX_GENERATIONS: Annotated[int, 'The maximum number of generations that the population could have'] \
        = 5
    MUTATION_PROBABILITY: Annotated[float, 'The probability that an individual could mutate'] \
        = 0.05

    individuals: Annotated[List[Individual], 'List of individuals']
    generations: Annotated[int, 'The number of generations of the population']
    generation_best_individuals: Annotated[List[Tuple[Individual, Individual]], 'The best individuals of every generation of the population. For every item in the List, the first value is the best individual by cost and the second value is the best individual by renewable generation']
    best_individuals: Annotated[Tuple[Individual, Individual], 'The best individuals of the population. The first value is the best individual by cost and the second value is the best individual by renewable generation']
    individuals_probabilities: Annotated[Tuple[List[float], List[float]], 'The probabilities of every individual to be selected. The first value is the probabilities by cost and the second value is the probabilities by renewable generation']

    def __init__(self, initial_individuals):
        self.individuals = initial_individuals
        self.generations = 0

        self.best_individuals = (None, None)
        self.generation_best_individuals = []
        self.calculate_generation_best_individuals()

        self.individuals_probabilities = []
        self.calculate_individuals_probabilities()

    def print(self, identation = '', print_individuals = False):
        print(f'Características de la Población:')
        print(f'Fitness de los mejores individuos: {self.best_individuals[0].fitness[0]} (Costo), {self.best_individuals[1].fitness[1]} (Generación Renovable)')
        print(f'Cantidad de Generaciones: {self.generations}')
        print(f'Probabilidades de los Individuos: {self.individuals_probabilities}')
        print(f'Cantidad de Individuos: {len(self.individuals)}')
        if print_individuals:
            for individual in self.individuals:
                individual.print()
        print('')
        
    def calculate_generation_best_individuals(self):
        self.generation_best_individuals.append( (\
            sorted(self.individuals, key=lambda ind: ind.fitness[0])[0], \
            sorted(self.individuals, key=lambda ind: ind.fitness[1])[-1]) )
        if self.best_individuals[0] is None:
            self.best_individuals = (self.generation_best_individuals[0][0], self.generation_best_individuals[0][1])
            return

        if self.best_individuals[0] is None or self.best_individuals[0].fitness[0] > self.generation_best_individuals[-1][0].fitness[0]:
            self.best_individuals = (self.generation_best_individuals[-1][0], self.best_individuals[1])
        if self.best_individuals[1] is None or self.best_individuals[1].fitness[1] < self.generation_best_individuals[-1][1].fitness[1]:
            self.best_individuals = (self.best_individuals[0], self.generation_best_individuals[-1][1])

    
    def calculate_individuals_probabilities(self):
        worse_cost = sorted(self.individuals, key=lambda ind: ind.fitness[0])[-1].fitness[0]
        worse_renewable_generation = sorted(self.individuals, key=lambda ind: ind.fitness[1])[-1].fitness[1]
        cost_sum, renewable_sum = sum([ind.fitness[0] for ind in self.individuals]) - worse_cost * len(self.individuals), \
            sum([ind.fitness[1] for ind in self.individuals]) - worse_renewable_generation * len(self.individuals)
        
        cost_sum = round(cost_sum, 8)
        renewable_sum = round(renewable_sum, 8)

        if cost_sum == 0:
            cost_probabilities = [1 / len(self.individuals) for _ in range(len(self.individuals))]
        else:
            cost_probabilities = [1 - (ind.fitness[0] - worse_cost) / cost_sum for ind in self.individuals]
        if renewable_sum == 0:
            renewable_probabilities = [1 / len(self.individuals) for _ in range(len(self.individuals))]
        else:
            renewable_probabilities = [(ind.fitness[1] - worse_renewable_generation) / renewable_sum for ind in self.individuals]

        self.individuals_probabilities = (cost_probabilities, renewable_probabilities)
    
    def get_best_fitness(self) -> (float, float):
        return self.best_individuals[0].fitness[0], self.best_individuals[1].fitness[1]
    
    def get_generation_fitness(self, generation_index) -> (float, float):
        return self.generation_best_individuals[generation_index][0].fitness[0], self.generation_best_individuals[generation_index][1].fitness[1]
        
    def get_all_fitness(self) -> List[Tuple[float, float]]: 
        return [self.get_generation_fitness(generation_index) for generation_index in range(self.generations)]
    
    def get_pareto_front(self) -> List[Individual]:
        pareto_front = []
        generation_bests = []
        for generation_best_individual in self.generation_best_individuals:
            generation_bests.append(generation_best_individual[0])
            generation_bests.append(generation_best_individual[1])

        for individual in generation_bests:
            is_dominated = False
            for other_individual in generation_bests:
                if other_individual != individual:
                    if (other_individual.fitness[0] <= individual.fitness[0] and other_individual.fitness[1] >= individual.fitness[1]) and \
                    (other_individual.fitness[0] < individual.fitness[0] or other_individual.fitness[1] > individual.fitness[1]):
                        is_dominated = True
                        break
            if not is_dominated:
                pareto_front.append(individual)
        return pareto_front

    def select_individual(self):
        is_cost = random.random() < 0.5
        probabilities = self.individuals_probabilities[0] if is_cost else self.individuals_probabilities[1]
        return random.choices(self.individuals, weights=probabilities)[0]

    def replace(self, min_generation_index):
        removing_indexes = []
        for ind_index, individual in enumerate(self.individuals):
            if min_generation_index > individual.generation_index and individual not in self.best_individuals:
                removing_indexes.append(ind_index)
        self.individuals = [individual for ind_index, individual in enumerate(self.individuals) if ind_index not in removing_indexes]
        
        selected_individuals = sorted(self.individuals, key=lambda ind: ind.fitness[0])[:250]
        selected_individuals.extend(sorted(self.individuals, key=lambda ind: ind.fitness[1])[-250:])
        
        self.individuals = selected_individuals

    def evolve(self):
        # Selection
        selected_individuals = [self.select_individual() for _ in range(int(len(self.individuals) * Population.SELECTION_FACTOR))]

        # Crossover
        new_individuals = []
        for _ in range(int(len(self.individuals) * Population.CROSSOVER_FACTOR)):
            firts_individual, second_individual = random.sample(selected_individuals, 2)
            new_individuals.extend(Individual.crossover(firts_individual, second_individual))

        # Mutation
        for individual in new_individuals:
            if random.random() < Population.MUTATION_PROBABILITY:
                individual.mutate()
        
        # Replace
        self.individuals.extend(new_individuals)
        self.calculate_generation_best_individuals()
        self.replace(self.generations - Population.MAX_GENERATIONS)
        self.calculate_individuals_probabilities()

        self.generations += 1
