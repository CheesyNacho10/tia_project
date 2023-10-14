from model import ProblemModelFactory, Poblation

POPULATION_SIZE = 100

if __name__ == '__main__':
    myProblemModel = ProblemModelFactory.create_random_problem_model()
    myPoblation = Poblation(myProblemModel, POPULATION_SIZE)
    last_best = myPoblation.best_individuals

    for i in range(100):
        myPoblation.evolve()
        if myPoblation.best_individuals != last_best:
            print(f'Generation {i}: {myPoblation.best_individuals}')
            last_best = myPoblation.best_individuals

    print(f'Best individual: {myPoblation.best_individuals}')
    print(f'Best fitness: {myPoblation.best_individuals.fitness}')
