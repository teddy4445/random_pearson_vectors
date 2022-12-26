# library imports
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Main:

    # CONSTS #
    POP_SIZE = 100
    GENERATIONS = 150
    MUTATION_RATE = 0.075
    ROYALTY_RATE = 0.05
    TOURNAMENT_SIZE = 10
    TOL = 0.02
    # END - CONSTS #

    def __init__(self):
        pass

    @staticmethod
    def run(vectors_size: int,
            a_range: tuple,
            b_range: tuple,
            target_pearson: float,
            csv_path: str,
            verbose: int = 0):
        """
        :param vectors_size: the size of the wanted vectors
        :param a_range: the range of the first vector
        :param b_range: the range of the second vector
        :param target_pearson: the wanted pearson between the vectors
        :param csv_path: the path to where we wish to save the result
        :param verbose: if 0 print nothing, if 1 prints status, if 2 prints graph
        :return: the csv
        """
        # generate populations
        a_delta = a_range[1] - a_range[0]
        b_delta = b_range[1] - b_range[0]
        solutions = [Solution.generate_random(vectors_size=vectors_size,
                                              a_start=a_range[0],
                                              a_delta=a_delta,
                                              b_start=b_range[0],
                                              b_delta=b_delta)
                     for _ in range(Main.POP_SIZE)]

        best_sol_so_far = solutions[0]
        for generation_index in range(Main.GENERATIONS):
            Main.mutate(solutions=solutions,
                        a_start=a_range[0],
                        a_delta=a_delta,
                        b_start=b_range[0],
                        b_delta=b_delta)
            Main.pop_cross_over(solutions=solutions)
            Main.fitness(solutions=solutions,
                         target_pearson=target_pearson)
            solutions = Main.tournament_with_royalty(solutions=solutions)
            best_sol_this_round = Solution.get_best_sol(solutions=solutions)

            if best_sol_this_round.fitness < best_sol_so_far.fitness:
                best_sol_so_far = best_sol_this_round

            # if print status
            if verbose >= 1:
                print("Gen = {} | Best fitness: {} (we want 0)".format(generation_index+1,
                                                                       best_sol_so_far.fitness))

            if best_sol_so_far.fitness < Main.TOL:
                if verbose >= 1:
                    print("We pre-stop due to the tolerance stop condition")
                break

        # end - tell user and save results
        if verbose >= 1:
            print("\nFinal solution obtains with pearson {:.4f} and asked {:.4f}".format(Solution.get_best_sol(solutions=solutions).pearson(),
                                                                                         target_pearson))

        # save the best
        best_gene = Solution.order(solutions=solutions)[0]
        best_gene.to_df().to_csv(csv_path,
                                 index=False)
        if verbose == 2:
            plt.scatter(x=best_gene.a_pop,
                        y=best_gene.b_pop,
                        color="blue")
            plt.show()
            plt.close()

    @staticmethod
    def fitness(solutions: list,
                target_pearson: float):
        [sol.sol_fitness(target_pearson=target_pearson)
         for sol in solutions]

    @staticmethod
    def mutate(solutions: list,
               a_start: float,
               a_delta: float,
               b_start: float,
               b_delta: float):
        [sol.mutate(a_start=a_start,
                    a_delta=a_delta,
                    b_start=b_start,
                    b_delta=b_delta)
         for sol in solutions]

    @staticmethod
    def pop_cross_over(solutions: list):
        # TODO: works without it, can be added later to make the algorithm converage faster
        pass

    @staticmethod
    def cross_over(c1: np.array,
                   c2: np.array):
        split_index = random.randint(0, len(c1)-1)
        new_c1 = c1[:split_index] + c2[split_index:]
        new_c2 = c2[:split_index] + c1[split_index:]
        return new_c1, new_c2

    @staticmethod
    def tournament_with_royalty(solutions: list):
        needed_size = len(solutions)
        solutions = Solution.order(solutions=solutions)
        new_solutions = solutions[:round(len(solutions)*Main.ROYALTY_RATE)].copy()
        delta_needed = needed_size - len(new_solutions)
        solutions = solutions[round(len(solutions) * Main.ROYALTY_RATE):]
        new_solutions.extend([Solution.get_best_sol(solutions=[solutions[int(random.random() * len(solutions))] for i in range(Main.TOURNAMENT_SIZE)])
                              for _ in range(delta_needed)])
        return new_solutions


class Solution:

    def __init__(self,
                 a_pop: np.array,
                 b_pop: np.array,
                 fitness: float):
        self.a_pop = a_pop
        self.b_pop = b_pop
        self.fitness = fitness

    def pearson(self):
        return np.corrcoef(self.a_pop, self.b_pop)[0][1]

    def sol_fitness(self,
                    target_pearson: float):
        self.fitness = abs(target_pearson - np.corrcoef(self.a_pop, self.b_pop)[0][1])

    def mutate(self,
               a_start: float,
               a_delta: float,
               b_start: float,
               b_delta: float):
        if random.random() < Main.MUTATION_RATE:
            if random.random() < 0.5:
                self.a_pop[random.randint(0, len(self.a_pop)-1)] = a_start + a_delta*random.random()
            else:
                self.b_pop[random.randint(0, len(self.b_pop)-1)] = b_start + b_delta*random.random()

    def to_df(self):
        return pd.DataFrame(data=np.transpose([self.a_pop,
                                               self.b_pop]),
                            columns=["a", "b"])

    @staticmethod
    def generate_random(vectors_size: int,
                        a_start: float,
                        a_delta: float,
                        b_start: float,
                        b_delta: float):
        return Solution(a_pop=[a_start + a_delta*random.random() for _ in range(vectors_size)],
                        b_pop=[b_start + b_delta*random.random() for _ in range(vectors_size)],
                        fitness=0)

    @staticmethod
    def get_best_fitness(solutions: list):
        return min([sol.fitness for sol in solutions])

    @staticmethod
    def get_best_sol(solutions: list):
        best_sol = solutions[0]
        for sol in solutions:
            if sol.fitness < best_sol.fitness:
                best_sol = sol
        return best_sol

    @staticmethod
    def order(solutions: list):
        return sorted(solutions, key=lambda s: s.fitness)

    def __repr__(self):
        return "Fitness = {:.4f}".format(self.fitness)

    def __str__(self):
        return " Fitness = {:.4f} --> A = {}, B = {}".format(self.fitness,
                                                             self.a_pop,
                                                             self.b_pop)


if __name__ == '__main__':
    Main.run(vectors_size=50,
             a_range=(30, 300),
             b_range=(18, 30),
             target_pearson=0.5,
             csv_path="result.csv",
             verbose=1)

