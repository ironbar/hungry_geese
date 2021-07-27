# Solution Summary
<!---https://www.kaggle.com/wiki/WinningModelDocumentationTemplate --->

## Solution summary

[Presentation slides](https://docs.google.com/presentation/d/1Qcf1tAl4PdOCoZuRddUdJ5ESJmevntuUf9S9Th_U8Aw/edit?usp=sharing)

### Key elements

- Deep Q* Learning
- Big model (~20M parameters)
- Simplified version of the game (just 3 movements, head centered and always looking north)
- League of agents approach with Elo rating
- Data augmentation on test (horizontal flip and adversary flip)

### Evolution of agent scores over time

![evolution of scores](res/2021-07-03-18-58-57.png)

- We can see that there is a pretty good correlation betweeen local and leaderboard score.
It is not perfect, but the relation is clear.
- Also we can see how wide the distribution of scores can be for the same agent
