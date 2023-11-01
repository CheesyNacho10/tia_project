import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
import model
import os
from datetime import datetime

class SimulatedAnnealingRunner:
    def __init__(self, initial_individual, initial_temperature, cooling_rate):
        self.current_individual = initial_individual
        self.current_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.pareto_front = [initial_individual]
        self.fitness_history = [(initial_individual.fitness[0], initial_individual.fitness[1])]
        self.directory_path = self.create_directory_structure()
        self.visualizer = SimulatedAnnealingVisualizer(self.directory_path)
        self.results_file_path = f'{self.directory_path}/simulated_annealing_results.txt'
        self.print_and_save_initial_information()

    def create_directory_structure(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        directory_path = f"simulated_annealing_results/{timestamp}"
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        return directory_path

    def print_and_save_initial_information(self):
        with open(self.results_file_path, 'w') as f:
            f.write('Características del Problema:\n')
            f.write(f'Cantidad de Slots: {len(self.current_individual.demands)}\n')
            f.write(f'Cantidad de Generadores: {len(self.current_individual.generators)}\n')
            f.write(f'\tCantidad de Generadores renovables: {len([generator for generator in self.current_individual.generators if generator.is_renewable])}\n')
            f.write(f'\tCantidad de Generadores no renovables: {len([generator for generator in self.current_individual.generators if not generator.is_renewable])}\n')
            max_generations_sum = [sum(generator.max_generations[i] for generator in self.current_individual.generators) for i in range(len(self.current_individual.demands))]
            f.write(f'\tGeneraciones Máximas: {max_generations_sum}\n')
            f.write(f'Demandas a cumplir: {self.current_individual.demands}\n')
            f.write('\nResultados por Iteración:\n')
            f.write('Iteración, Costo, Generación Renovable\n')

    def print_and_save_iteration_results(self, iteration):
        with open(self.results_file_path, 'a') as f:
            f.write(f'{iteration}, {self.current_individual.fitness[0]}, {self.current_individual.fitness[1]}\n')

    def dominates(self, individual_a, individual_b):
        better_in_all = (individual_a.fitness[0] <= individual_b.fitness[0] and
                         individual_a.fitness[1] >= individual_b.fitness[1])
        better_in_one = (individual_a.fitness[0] < individual_b.fitness[0] or
                         individual_a.fitness[1] > individual_b.fitness[1])
        return better_in_all and better_in_one

    def cooling_function(self):
        return self.current_temperature * self.cooling_rate

    def run(self):
        try:
            iteration = 0
            while self.current_temperature > 1e-4:
                neighbor = self.current_individual.get_neighbor()
                
                if self.dominates(neighbor, self.current_individual):
                    self.current_individual = neighbor
                    self.pareto_front = [ind for ind in self.pareto_front if not self.dominates(neighbor, ind)]
                    if not any(self.dominates(ind, neighbor) for ind in self.pareto_front):
                        self.pareto_front.append(neighbor)
                elif not any(self.dominates(ind, neighbor) for ind in self.pareto_front):
                    self.pareto_front.append(neighbor)
                    self.pareto_front = [ind for ind in self.pareto_front if not self.dominates(neighbor, ind) or ind is neighbor]

                self.current_temperature = self.cooling_function()
                self.fitness_history.append((self.current_individual.fitness[0], self.current_individual.fitness[1]))
                self.print_and_save_iteration_results(iteration)
                iteration += 1
            
            self.visualizer.graph_fitness(self.fitness_history)
            self.visualizer.graph_pareto_front(self.pareto_front)
            self.visualizer.graph_fitness_distribution(self.fitness_history)

            for i, ind in enumerate(self.pareto_front, start=1):
                print(f'Individuo {i} en el Frente de Pareto: Costo = {ind.fitness[0]}, Generación Renovable = {ind.fitness[1]}')

        except KeyboardInterrupt:
            print('\nEjecución interrumpida por el usuario.')
            self.visualizer.graph_fitness(self.fitness_history)
            self.visualizer.graph_pareto_front(self.pareto_front)
            self.visualizer.graph_fitness_distribution(self.fitness_history)

            for i, ind in enumerate(self.pareto_front, start=1):
                print(f'Individuo {i} en el Frente de Pareto: Costo = {ind.fitness[0]}, Generación Renovable = {ind.fitness[1]}')

class SimulatedAnnealingVisualizer:
    def __init__(self, directory_path):
        self.directory_path = directory_path

    def graph_fitness(self, fitness_history):
        fig, ax = plt.subplots()
        ax.plot([fitness[0] for fitness in fitness_history], label='Costo (Minimizar)')
        ax.plot([fitness[1] for fitness in fitness_history], label='Generación Renovable (Maximizar)')
        ax.set_xlabel('Iteración')
        ax.set_ylabel('Fitness')
        ax.legend()
        plt.savefig(f'{self.directory_path}/simulated_annealing_fitness.png')
        plt.close(fig)

    def graph_pareto_front(self, pareto_front):
        fig, ax = plt.subplots()
        pareto_points = sorted([(ind.fitness[0], ind.fitness[1]) for ind in pareto_front], key=lambda x: x[0])
        
        # Eliminar duplicados basándonos en el costo
        pareto_points = sorted(set(pareto_points), key=lambda x: x[0])
        
        x, y = zip(*pareto_points)
        
        if len(x) > 3:
            xnew = np.linspace(min(x), max(x), 300)
            spl = make_interp_spline(x, y, k=3)
            ynew = spl(xnew)
            ax.plot(xnew, ynew, 'r-')
        elif len(x) > 1:
            ax.plot(x, y, 'r-')
        elif len(x) == 1:
            ax.scatter(x, y)
        
        ax.scatter(x, y)
        ax.set_xlabel('Costo (Minimizar)')
        ax.set_ylabel('Generación Renovable (Maximizar)')
        ax.set_title('Frontera de Pareto')
        plt.savefig(f'{self.directory_path}/pareto_front.png')
        plt.close(fig)

    def graph_fitness_distribution(self, fitness_history):
        costs, renewables = zip(*fitness_history)
        fig, axs = plt.subplots(2, figsize=(10, 10))
        
        axs[0].hist(costs, bins=20)
        axs[0].set_title('Distribución de Costos')
        axs[0].set_xlabel('Costo')
        axs[0].set_ylabel('Frecuencia')

        axs[1].hist(renewables, bins=20)
        axs[1].set_title('Distribución de Generación Renovable')
        axs[1].set_xlabel('Generación Renovable')
        axs[1].set_ylabel('Frecuencia')

        plt.tight_layout()
        plt.savefig(f'{self.directory_path}/fitness_distribution.png')
        plt.close(fig)

def main():
    initial_temperature = 1000.0
    cooling_rate = 0.9995
    initial_individual = model.IndividualFactory.create_random_individual()
    runner = SimulatedAnnealingRunner(initial_individual, initial_temperature, cooling_rate)
    runner.run()

if __name__ == '__main__':
    for _ in range(100):
        main()
