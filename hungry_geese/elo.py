import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class EloRanking():
    def __init__(self, initial_ranking, agents_to_update, k=8, verbose=False):
        """
        Computes Elo ranking

        Parameters
        -----------
        agents : list of str
            Name of the agents that will be used in the ranking
        agents_to_update  : set of str
            Name of the agents that can be updated, agents not in the list will have fixed score
        k : float
            Elo speed constant, it seems that 8 is ok for our problem
        verbose : bool
        """
        self.ranking = initial_ranking
        self.agents_to_update = agents_to_update
        self.k = k
        self.verbose = verbose

    def add_match(self, agents, scores):
        updates = self._compute_updates(agents, scores)
        if self.verbose: print(updates)
        self._apply_updates(updates)

    def _compute_updates(self, agents, scores):
        updates = {agent:0 for agent in np.unique(agents)}
        for idx, agent1 in enumerate(agents):
            score1 = scores[idx]
            for agent2, score2 in zip(agents[idx+1:], scores[idx+1:]):
                if agent1 == agent2:
                    continue
                result = elo_result(score1, score2)
                update = self.k*(result - elo_expectation(self.ranking[agent1][-1], self.ranking[agent2][-1]))
                if agent1 in self.agents_to_update: updates[agent1] += update
                if agent2 in self.agents_to_update: updates[agent2] -= update
                self.verbose: print(agent1, score1, agent2, score2, elo_result(score1, score2))
        return updates

    def _apply_updates(self, updates):
        for agent in self.ranking:
            if agent in updates:
                self.ranking[agent].append(self.ranking[agent][-1] + updates[agent])
            else:
                self.ranking[agent].append(self.ranking[agent][-1])

    def plot(self):
        for agent, ranking in self.ranking.items():
            #plt.plot(ranking, label=agent, marker='o')
            plt.plot(ranking, label=agent)
        plt.legend(loc=0)
        plt.grid(axis='y')

    def summary(self):
        agents = list(self.ranking.keys())
        ranking = [int(self.ranking[agent][-1]) for agent in agents]
        summary = pd.DataFrame(dict(ranking=ranking), index=agents)
        return summary.sort_values('ranking', ascending=False)


def elo_result(score1, score2):
    if score1 > score2:
        return 1
    elif score1 == score2:
        return 0.5
    else:
        return 0

def elo_expectation(ranking1, ranking2):
    return 1./(1+10**((ranking2 - ranking1)/400))