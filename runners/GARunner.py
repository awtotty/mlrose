try:
    import mlrose
except:
    import sys
    sys.path.append("..")
    import mlrose

from runners._RunnerBase import RunnerBase


class GARunner(RunnerBase):

    def __init__(self, problem, seed, iteration_list, population_sizes, mutation_rates, max_attempts=500,
                 generate_curves=True):
        super().__init__(problem, seed, iteration_list, max_attempts, generate_curves)
        self.population_sizes = population_sizes
        self.mutation_rates = mutation_rates

    def run(self):
        return super()._run_experiment(name='GA',
                                       algorithm=mlrose.genetic_alg,
                                       pop_size=('Population Size', self.population_sizes),
                                       mutation_prob=('Mutation Rate', self.mutation_rates))



