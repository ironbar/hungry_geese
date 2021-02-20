# Data Understanding

## Collect initial data

In this channel there is no data, instead there is a [game environment](https://github.com/Kaggle/kaggle-environments)
where we can train agents and there are two available agents: random and greedy.

## External data

<!--- It is allowed in this challenge? If so write it here ideas of how to find
it and if people have already posted it on the forum describe it. --->

There is a [notebook](https://www.kaggle.com/robga/simulations-episode-scraper-match-downloader)
that allows to download games from Kaggle Leaderboard, it could be interesting to try offline learning
with those games.

## Describe data

<!---Describe the data that has been acquired, including the format of the data,
the quantity of data (for example, the number of records and fields in each table),
the identities of the fields, and any other surface features which have been
discovered. Evaluate whether the data acquired satisfies the relevant requirements. --->

There is an [schema](https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/hungry_geese/hungry_geese.json)
that describes the information pretty well.

```python
[{'action': 'SOUTH', # previous action, ignore it on start of the game
  'reward': 10112,
  'info': {},
  'observation': {'remainingOverageTime': 60, # this could be useful to allocate resources
   'step': 100, # step idx, starts at 0
   'geese': [[53, 42, 31, 20, 19, 18, 17, 16, 15, 4, 3, 14], #A position of a goose with the 0 position representing the head.
    [],
    [],
    [10, 0, 1, 12, 23, 34, 45, 56, 67, 68, 2, 13, 24]],
   'food': [63, 33], #"Positions on the board where food is present."
   'index': 0 #Index of the current agent's goose in the list of geese.
   },
  'status': 'ACTIVE' # status of the agent: ACTIVE or DONE
  },
 {'action': 'NORTH',
  'reward': 7104,
  'info': {},
  'observation': {'remainingOverageTime': 60, 'index': 1},
  'status': 'DONE'},
 {'action': 'NORTH',
  'reward': 9114,
  'info': {},
  'observation': {'remainingOverageTime': 60, 'index': 2},
  'status': 'DONE'},
 {'action': 'WEST',
  'reward': 10113,
  'info': {},
  'observation': {'remainingOverageTime': 60, 'index': 3},
  'status': 'ACTIVE'}]
```

I think that we have to transform both the state and the reward before feeding it to the network.

- The state is very compact and allows to do transitions very easy (we only have to move head and tail),
however it is not trivial to extract 2d information from there. So I have the intuition that I should
give the board to the agent, not the compact state.
- The reward is very informative, but what matters is the ranking. So I think it would be much
clearer to use the ranking on the match as the reward

## Explore data

<!---This task addresses data mining questions using querying, visualization,
and reporting techniques. These include distribution of key attributes (for example,
the target attribute of a prediction task) relationships between pairs or small
numbers of attributes, results of simple aggregations, properties of significant
sub-populations, and simple statistical analyses.

Some techniques:
* Features and their importance
* Clustering
* Train/test data distribution
* Intuitions about the data
--->

## Verify data quality

<!---Examine the quality of the data, addressing questions such as: Is the data
complete (does it cover all the cases required)? Is it correct, or does it contain
errors and, if there are errors, how common are they? Are there missing values in
the data? If so, how are they represented, where do they occur, and how common are they? --->

## Amount of data

<!---
How big is the train dataset? How compared to the test set?
Is enough for DL?
--->
