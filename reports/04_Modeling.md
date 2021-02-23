# Modeling

## Select modeling technique

<!---Document the actual modeling technique that is to be used. If multiple
techniques are applied, perform this task separately for each technique.
Many modeling techniques make specific assumptions about the data—for example,
that all attributes have uniform distributions, no missing values allowed,
class attribute must be symbolic, etc. Record any such assumptions made. --->

I will be using Reinforcement Learning techniques in this challlenge.

## Generate test design

<!---Describe the intended plan for training, testing, and evaluating the models.
A primary component of the plan is determining how to divide the available dataset
into training, test, and validation datasets.

Doing a plot of score vs train size could be helpful to decide the validation strategy

Depending on the size of the data we have to decide how we are going to use submissions.
The less the submissions the most confidence we can have on the score. However sometimes
the data distribution is very different, or the size of the data is small and we have
to make a lot of submissions. Sometimes is not easy to have a good correlation between
validation score and LB score
--->

I have decided to implement an internal league of agents to measure progress. Ideally I could train
my models completely isolated from Kaggle and submit a single perfect agent to the leaderboard.
That would be very epic.

![alphastar league](res/2021-02-20-10-24-50.png)

### Theory about ranking

[Youtube Ranking Systems: Elo, TrueSkill and Your Own](https://www.youtube.com/watch?v=VnOVLBbYlU0)

#### [Elo rating system](https://en.wikipedia.org/wiki/Elo_rating_system)

Having a score 400 points higher means that you will win 90% of the times. If the number of matches is big enough the players will reach equilibrium.

In our case Agents are stationary, they do not improve over time. So I only need to find the relative position to existing agents.

Problems could arise if a new agent is not able to win all the previous agents. In that case the
rating will not converge unless the match distribution is fixed. So I have to create an environment
where each new agent is trained against all previous ones and learns to beat them all.

Some links:

- [Elo sucks — better multiplayer rating systems for smaller games](https://medium.com/acolytefight/elo-sucks-better-multiplayer-rating-systems-for-smaller-games-8ca588ee652f)
- [Multiplayer Elo](http://www.tckerrigan.com/Misc/Multiplayer_Elo)

#### [TrueSkill](https://en.wikipedia.org/wiki/TrueSkill)

TrueSkill is more complex and powerful than Elo, and I think it is used on Kaggle or something similar.

However I think Elo is enough for my problem and I will use it.

### Ranking definition

- I will use Elo ranking
- Greedy agent will have a fixed score of 1000
- I will train a first model that is able to beat the greedy agent consistently, and then another one
that is able to beat agents 1 and 2, and so on
- Old agent scores will be fixed, and I will just compute the ranking for the new agent
- New agents will be initialized with the ranking of the best existing agent
- To adapt to multiplayer Elo I will consider that on each match the agent has won over agents with
lower score and has lose over agents with greater score

## Iteration 1. Hand coded agents

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.
--->

### Goal

On a first step I'm going to develop or download from the forum hand-coded agents. This will
allow to implement the ranking system.

### Links

### Development

#### Agents from the forum

- https://www.kaggle.com/ilialar/risk-averse-greedy-goose
- https://www.kaggle.com/ilialar/goose-on-a-healthy-diet
- https://www.kaggle.com/ihelon/hungry-geese-agents-comparison
- https://www.kaggle.com/leontkh/hard-coded-passive-aggressive-agent

#### My agent

I would like to create an agent that only grows if there are bigger agents than him. I will only
implement it if it is not already implemented on the forum

#### Championship

I'm going to run matches between all the agents by selecting them randomly and use the results
to compute the first Elo ranking. I will check that at least there are two different agents
on each match.

I could play with different speeds for the Elo rating and see the effect.

### Results


## Iteration 2. Deep Q Learning

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.
--->

### Goal

I'm going to start with the simplest powerful approach that I can think of: Deep Q learning. On this
iteration I would like to see how far a model trained this way can go and how long does it take to
learn.

### Links

https://www.kaggle.com/victordelafuente/dqn-goose-with-stable-baselines3-pytorch#Using-our-custom-environment

### Development

### Results
