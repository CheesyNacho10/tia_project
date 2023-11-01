import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from scipy.interpolate import make_interp_spline
import model

class GeneticAlgorithmVisualizer:
    def __init__(self, population: model.Population, directory_path: str):
        self.population = population
        self.directory_path = directory_path

    def graph_fitness(self):
        fig, ax = plt.subplots()
        fitness_history = self.population.get_all_fitness()
        ax.plot([fitness[0] for fitness in fitness_history], label='Costo')
        ax.plot([fitness[1] for fitness in fitness_history], label='Generación Renovable')
        ax.set_xlabel('Generación')
        ax.set_ylabel('Fitness')
        ax.legend()
        plt.savefig(f'{self.directory_path}/genetic_algorithm_fitness.png')
        plt.close(fig)

    def graph_pareto_front(self):
        fig, ax = plt.subplots()
        pareto_front = self.population.get_pareto_front()
        pareto_points = sorted([(ind.fitness[0], ind.fitness[1]) for ind in pareto_front], key=lambda x: x[0])
        x, y = zip(*sorted(set(pareto_points), key=lambda x: x[0]))

        if len(x) > 3:
            xnew = np.linspace(min(x), max(x), 300)
            spl = make_interp_spline(x, y, k=3)
            ynew = spl(xnew)
            ax.plot(xnew, ynew, 'r-')
        elif len(x) > 1:
            ax.plot(x, y, 'r-')

        ax.scatter(x, y)
        ax.set_xlabel('Costo')
        ax.set_ylabel('Generación Renovable')
        ax.set_title('Frontera de Pareto')
        plt.savefig(f'{self.directory_path}/pareto_front.png')
        plt.close(fig)

    def graph_fitness_distribution(self):
        last_generation_fitness = [(ind.fitness[0], ind.fitness[1]) for ind in self.population.individuals]
        costs, renewables = zip(*last_generation_fitness)
        fig, axs = plt.subplots(2, figsize=(10, 10))
        axs[0].hist(costs, bins=20)
        axs[0].set_title('Distribución de Costos')
        axs[1].hist(renewables, bins=20)
        axs[1].set_title('Distribución de Generación Renovable')
        plt.savefig(f'{self.directory_path}/fitness_distribution.png')
        plt.close(fig)

class GeneticAlgorithmRunner:
    def __init__(self, population_size: int):
        self.population_size = population_size
        self.population = model.PopulationFactory.create_random_population(population_size)
        self.directory_path = self.create_directory_structure()
        self.visualizer = GeneticAlgorithmVisualizer(self.population, self.directory_path)
        self.last_best_cost, self.last_best_renewable = self.population.get_generation_fitness(self.population.generations)
        self.generations_without_improvement = 0
        self.iterations_for_saving = 10

    def run(self):
        self.print_and_save_initial_information()
        try:
            while self.generations_without_improvement < len(self.population.individuals) * 10:
                self.perform_evolution_step()
                if self.population.generations % self.iterations_for_saving == 0:
                    self.visualizer.graph_fitness()
                    self.visualizer.graph_fitness_distribution()
            print(f'Finalizado en la generación {self.population.generations} con {len(self.population.individuals)} individuos.')
            self.visualizer.graph_pareto_front()
            self.visualizer.graph_fitness()
        except KeyboardInterrupt:
            print('\nEjecución interrumpida por el usuario.')
            self.visualizer.graph_pareto_front()
            self.visualizer.graph_fitness()
            self.visualizer.graph_fitness_distribution()

    def create_directory_structure(self) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        directory_path = f"genetic_algorithm_results/{timestamp}"
        os.makedirs(directory_path, exist_ok=True)
        return directory_path

    def print_and_save_initial_information(self):
        print(f'Demanda: {self.population.individuals[0].demands}')
        with open(f'{self.directory_path}/genetic_algorithm_model.txt', 'w') as f:
            f.write('Características del Problema:\n')
            f.write(f'Cantidad de Individuos: {self.population_size}\n')
            f.write(f'Cantidad de Slots: {len(self.population.individuals[0].demands)}\n')
            f.write(f'Cantidad de Generadores: {len(self.population.individuals[0].generators)}\n')
            f.write(f'\tCantidad de Generadores renovables: {len([generator for generator in self.population.individuals[0].generators if generator.is_renewable])}\n')
            f.write(f'\tCantidad de Generadores no renovables: {len([generator for generator in self.population.individuals[0].generators if not generator.is_renewable])}\n')
            individual = self.population.individuals[0]
            max_generations_sum = [sum(generator.max_generations[i] for generator in individual.generators) for i in range(len(individual.demands))]
            f.write(f'\tGeneraciones Máximas: {max_generations_sum}\n')
            f.write(f'Demandas a cumplir: {individual.demands}\n')
        with open(f'{self.directory_path}/genetic_algorithm_results.txt', 'w') as f:
            f.write('Resultados por Generación:\n')
            f.write('Generación, Mejor Costo, Mejor Generación Renovable, Mejor Costo (%), Mejor Generación Renovable (%)\n')

    def perform_evolution_step(self):
        best_cost, best_renewable = self.population.get_generation_fitness(self.population.generations)
        cost_improvement = (best_cost - self.last_best_cost) / self.last_best_cost * 100 if self.last_best_cost != 0 else 0
        renewable_improvement = (best_renewable - self.last_best_renewable) / self.last_best_renewable * 100 if self.last_best_renewable != 0 else 0
        with open(f'{self.directory_path}/genetic_algorithm_results.txt', 'a') as f:
            f.write(f'{self.population.generations}, {best_cost}, {best_renewable}, {cost_improvement}, {renewable_improvement}\n')
        print(f'Generación {self.population.generations} con {len(self.population.individuals)} individuos: Mejor Costo = {best_cost}, Mejor Generación Renovable = {best_renewable}')
        print(f'Mejor costo = {self.population.best_individuals[0].fitness[0]}, Mejor generación de renovable = {self.population.best_individuals[1].fitness[1]}')
        print(f'Costo Mejorado en {cost_improvement}%, Generación Renovable Mejorada en {renewable_improvement}%\n')
        if abs(cost_improvement) < 0.01 and abs(renewable_improvement) < 0.01:
            self.generations_without_improvement += 1
        else:
            self.generations_without_improvement = 0
        self.last_best_cost, self.last_best_renewable = best_cost, best_renewable
        self.population.evolve()

if __name__ == '__main__':
    for _ in range(100):
        runner = GeneticAlgorithmRunner(1000)
        runner.run()
