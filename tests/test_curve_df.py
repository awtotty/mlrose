try:
    import mlrose_hiive
except:
    import sys
    sys.path.append("..")

import unittest
import numpy as np
from mlrose_hiive import (GARunner, RHCRunner, MIMICRunner, SARunner, DiscreteOpt, FourPeaks)

class TestCurveDF(unittest.TestCase): 

    @staticmethod
    def test_rhc(self): 
        fitness = FourPeaks()
        problem = DiscreteOpt(length=10, fitness_fn=fitness, maximize=True, max_val=2)
        iters_list = [0,10]
        restart_list = [0,10]

        runner = RHCRunner(
            problem=problem, 
            experiment_name="test curve rhc", 
            seed=1, 
            iteration_list=iters_list,
            restart_list=restart_list,
            max_attempts=10, 
        )
        _, df_curves = runner.run()

        runs = [df_curves.loc[df_curves["Restarts"] == r] for r in restart_list]

        for run in runs: 
            assert 0 == run["Iteration"].array[0], "First iteration of results is not 0."

    @staticmethod
    def test_sa(self): 
        fitness = FourPeaks()
        problem = DiscreteOpt(length=10, fitness_fn=fitness, maximize=True, max_val=2)
        iters_list = [0,10]
        temperature_list = [1, 10] 

        runner = SARunner(
            problem=problem, 
            experiment_name="test curve sa", 
            seed=1, 
            iteration_list=iters_list,
            temperature_list=temperature_list,
            max_attempts=10, 
        )
        _, df_curves = runner.run()

        runs = [df_curves.loc[df_curves["schedule_init_temp"] == t] for t in temperature_list]

        for run in runs: 
            assert 0 == run["Iteration"].array[0], "First iteration of results is not 0."


    @staticmethod
    def test_ga(self): 
        fitness = FourPeaks()
        problem = DiscreteOpt(length=10, fitness_fn=fitness, maximize=True, max_val=2)
        iters_list = [0,10]
        population_sizes = [1, 10] 

        runner = GARunner(
            problem=problem, 
            experiment_name="test curve ga", 
            seed=1, 
            iteration_list=iters_list,
            population_sizes=population_sizes, 
            max_attempts=10, 
        )
        _, df_curves = runner.run()

        runs = [df_curves.loc[df_curves["Population Size"] == p] for p in population_sizes]

        for run in runs: 
            assert 0 == run["Iteration"].array[0], "First iteration of results is not 0."

    @staticmethod
    def test_mimic(self): 
        fitness = FourPeaks()
        problem = DiscreteOpt(length=10, fitness_fn=fitness, maximize=True, max_val=2)
        iters_list = [0,10]
        population_sizes = [1, 10] 

        runner = MIMICRunner(
            problem=problem, 
            experiment_name="test curve mimic", 
            seed=1, 
            iteration_list=iters_list,
            population_sizes=population_sizes, 
            max_attempts=10, 
        )
        _, df_curves = runner.run()

        runs = [df_curves.loc[df_curves["Population Size"] == p] for p in population_sizes]

        for run in runs: 
            assert 0 == run["Iteration"].array[0], "First iteration of results is not 0."