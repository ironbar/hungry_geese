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
- I will compute ranking for hard-coded agents first, and I won't update those scores since I expect
that the RL agents will be better
- I will train a first model that is able to beat the greedy agent consistently, and then another one
that is able to beat agents 1 and 2, and so on
- Old agent scores will be fixed, and I will just compute the ranking for the new agent
- New agents will be initialized with the ranking of the best existing agent
- To adapt to multiplayer Elo I will consider that on each match the agent has won over agents with
lower score and has lose over agents with greater score

## Iteration 1. Hard coded agents

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.
--->

### Goal

On a first step I'm going to develop or download from the forum hard-coded agents. This will
allow to implement the ranking system.

### Development

#### Agents from the forum

- https://www.kaggle.com/ilialar/risk-averse-greedy-goose
- https://www.kaggle.com/ilialar/goose-on-a-healthy-diet
- https://www.kaggle.com/ihelon/hungry-geese-agents-comparison
- https://www.kaggle.com/leontkh/hard-coded-passive-aggressive-agent

#### Championship

I'm going to run matches between all the agents by selecting them randomly and use the results
to compute the first Elo ranking. I will check that at least there are two different agents
on each match.

I could play with different speeds for the Elo rating and see the effect.

### Results

| model                   | ranking |
|-------------------------|---------|
| boilergoose             | 1269    |
| besthoarder             | 1227    |
| crazy_goose             | 1186    |
| risk_averse_goose       | 1176    |
| goose_on_a_healthy_diet | 1157    |
| straightforward_bfs     | 1004    |
| greedy                  | 981     |
| greedyhoarder           | 766     |
| random_plus             | 706     |
| random                  | 523     |

So this is very interesting, we have compared many hard-coded agents and see that there are much stronger agents than greedy. For example boilergoose will beat greedy agent 83% of the times.

This gives us a very good start point to train reinforcement learning agents. We should aim to create a first RL agent that scores around 1500. Then aim to 1700, 1900... and create a ladder of improving agents.

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

### Game interface

In order to be able to train an agent I have to first create a game interface that provides the information
in the best way for our model.

- Transform the board into a 3d tensor
- Remember last movements
- Compute useful features
- Save the whole episode for later training

### Study Kaggle Environments library

Let's first have a look at the library to better undertand how to deal with it.

There is an [OpenAI Gym interface](https://github.com/Kaggle/kaggle-environments#training) that may
be useful to simplify training.

The other option is to use [step](https://github.com/Kaggle/kaggle-environments#stepping) on the environment.
However this requires to compute the action for all the agents.

### Development

#### Design

```bash
Agent
    Model
    State or GameInterface
        update(observation, configuration)
        render(state)
        history()
        reset()
```

Do everything with tests.

I could pretrain the Q function using matches from the best hard-coded agent.

Start with a single processor, leave the process of paralellization for the future.

#### Playing against random agents

To verify that the model is able to learn I'm going to play against random agents. It should be
easy to learn to beat them. I have been able to train a model that apparently beats the random
agent consistently, however it is struggling to beat the greedy player.

Let's evaluate the agents to verify this.

We have been able to train a model that is better than random. When it plays against 3 random
agents it beats them consistently, however that's not the case when more than one q value agents
are used, which suggest collisions.

This raises concerns also about evaluating agents when there is no upper bound. We get very
optmistic scores.

The truth ranking is around 866 points compared to 706 by random pluss and 981 by greedy.

![q value learning against random agents](res/2021-03-13-08-39-03.png)

#### Playing against greedy agents

First experiments reveal a score almost identical to the greedy agent, however when using more than
one agent the score drops to 905 compared to 981 of the greedy agent.

![q value learning against greedy agents](res/2021-03-13-08-41-16.png)

#### Pretrain the agent on random agent games

The bigger the train dataset the better, but gains are diminishing.

By pretraining the model on this data the agent is able to achieve an impressive elo score of around
1140, much better than random (706) and greedy (981). So we were able to outperform the iterative
q value learning simply by pretraining.

![pretrained agent on random agent games](res/2021-03-13-08-56-42.png)

#### Pretrain the agent on epsilon-greedy or greedy agent games

After the success of pretraining on random agents it seemed logical to try with greedy or epsilon-greedy
agents pretrain. However the resulting agents are no better than the greedy agent.

My explanation for this behaviour is that the Q value function becomes too complex and the model is
unable to learn it.

![comparison of losses](res/2021-03-13-09-02-04.png)

In the plot above it is seen that the loss achieved when pretraining on random agent games is
around 1.32, while for epsilon greedy agent game is 1.66 and 1.68 for greedy games. This may
suggest that the more advanced the agent the more complex the Q value function.

I have tried increasing the agent capacity but there was overfit to the train set.

### Results

- It is hard to train a model to learn the Q value function, learning rate is critical
- The Q function is not easy. It depends on the agents we are facing and on the policy of our agent.
Moreover learning the reward function is not easy, because we have to predict what the other agents
are going to do. So this is an argument in favour of learning a policy instead of a value function
- We have been able to train an agent to become better than the random agent
- We have been able to pretrain an agent on random agents game to become better than greedy agent
- We have discovered that current Q value function is very complex and difficult to learn

### Arguments against Q value function

At the start of the game there is no information to predict the Q value function.

![undefinied Q value function at the start](res/2021-03-13-09-25-33.png)

I sugggest to update the reward to be the ranking of the goose on the next step. I have to better
think about propagation of the information backwards.

## Iteration 3. Exploring other reward functions

### Goal

The goal is to see if using other reward functions may simplify the Q value function and thus enable
learning from agents such as the greedy agent.

### Development

#### Current reward function

Let's remember how the current reward function is implemented:

- If the step is not terminal the reward is zero except some of the other geese has died in the previous
step. For each dead goose the reward is 1.
- If the step is terminal then if the goose has died the reward is -1, otherwise it gets 1 for each
living smaller goose and 0.5 for living goose of the same size.

That is the definition of the reward for each state. The cumulative reward is computed by summing
all the received rewards without any discount factor. The maximum cumulative reward is 3, and the minimun
is -1.

The problems of this reward function are:

- The relation between the state of the game and the reward is undefined. For example at the start of
the game there is no way to know which agent will win so there is no option to predict the reward.
- The problem above leads to have a very big upper bound when learning the q value function

The advantages of this reward function are:

- Maximizing this reward function ensures that we maximize the expected score on the leaderboard
- Reward is given when other goose die, which encourages to be a goose killer

#### Alternative reward function proposal

To solve the problem of being undefined I propose the following reward

- If the step is not terminal give a reward that is the current ranking of the agent. Give a reward
of 1 for each smaller goose and 0.5 for each goose of the same size
- If the step is terminal and the agent dies then give a negative reward, otherwise return the ranking
as defined above

If I compute the cumulative reward by summing without discount factor then again I will have an undefined
Q value function, since the result of the episode is unknown. Instead I propose to use a discount factor,
or even better a moving average. The advantage of the moving average is that we get good bounds for the
cumulative reward, which does not happen when using discount factor. The size of the window will be a
parameter to tune, the bigger the size of the window the more the model will have to look to the future.

Advantages of this approach:

- If we use a window of 1 (do not look into the future) then the q value function is much clearly defined.
Of course there is some uncertainty because movements of the other players are unknown, but it is a much
more simpler function than the previous one
- There are still incentives to kill other gooses if they have the same size or bigger than us
- There are incentives to be bigger than the other gooses, but not to be much bigger
- Maximizing this function also leads to maximizing leaderboard score

Problems of this approach:

- The more we look into the future the more complex the function becomes, we could probably probe this
experimentally

#### Implementation

Since it is possible that I could develop new reward functions in the future, I think I should make
a function that receives a keyword and computes reward using that keyword. This will take a little
bit of more work today, but will save time in the future.

I will call the new reward the `ranking_reward`, and the previous existing reward the `sparse_reward`.

The new `ranking_reward` has two parameters:

- Reward when the agent dies
- Window for averaging rewards (backwards averaging)

I'm going to supply both parameters in the name, for example `ranking_reward_-1_2` means the reward when dying will be -1 and the window for averaging will be 2. They are probably related.

#### Pretraining on random agents game

Let's start by pretraining on random agents game to see how different the training loss and the agent's performance is when training on this loss.

| model                                                     | elo score | validation loss |
|-----------------------------------------------------------|-----------|-----------------|
| Baseline_sparse_reward_80000                              | 1140      | ~1.3            |
| 03_ranking_reward_-1_1_40000                              | 995       | 0.13            |
| 04_ranking_reward_-1_2_10000episodes                      | 1102      | 0.18            |
| 05_ranking_reward_-1_3_10000episodes                      | 1109      | 0.20            |
| 06_ranking_reward_-1_4_10000episodes                      | 1071      | 0.23            |
| 08_ranking_reward_-1_4_10000episodes_encoder_x32          | 1002      | 0.28            |
| 07_ranking_reward_-1_5_10000episodes                      | 992       | 0.25            |
| 09_ranking_reward_-2_5_10000episodes                      | 1102      | 0.39            |
| 10_ranking_reward_-4_5_10000episodes                      | 1138      | 0.79            |
| 11_ranking_reward_-4_5_80000episodes                      | 1116      | 0.751           |
| 12_ranking_reward_-4_5_80000episodes_reduce_lr_on_plateau | 1230      | 0.747           |

- The loss of the new reward function is much smaller compared to sparse reward baseline, the direct comparison is 1.3 vs 0.13
- Clearly using only 1 step lookahead is not optimal
- The number of steps looakhead and the dead reward is related, as shown by the experiments with 5 steps.
- The bigger the lookahead steps the more complex the function as shown by the validation loss
- The model with reduce lr on plateau achieves the best elo score, although training loss is similar

So this experiments probe that the loss is much smaller with the new reward, and we were able to train
a better agent than the baseline.

#### Pretraining on greedy or epsilon greedy games

| agent               | model               | elo score |
|---------------------|---------------------|-----------|
| greedy              | -                   | 981       |
| greedy              | ranking_reward_-1_3 | 1023      |
| greedy              | ranking_reward_-2_3 | 1083      |
| greedy              | ranking_reward_-4_3 | 1068      |
| epsilon 0.05 greedy | ranking_reward_-1_3 | 1121      |
| epsilon 0.05 greedy | ranking_reward_-2_3 | 1095      |
| epsilon 0.05 greedy | ranking_reward_-4_3 | 1087      |

We were able to improve over greedy agent score. It seems that epsilon greedy playing helps slightly.
With the previous reward we were unable to do this.

#### Pretraining on boilergoose games

So now we want to try to improve over the current best agent: boilergoose.

| agent                    | model               | elo score |
|--------------------------|---------------------|-----------|
| boilergoose              | -                   | 1269      |
| boilergoose              | ranking_reward_-2_3 | 744       |
| boilergoose 0.05 epsilon | ranking_reward_-2_3 | 1021      |
| boilergoose 0.02 epsilon | ranking_reward_-2_3 | 1019      |
| boilergoose 0.01 epsilon | ranking_reward_-2_3 | 1055      |
| combo1                   | ranking_reward_-2_3 | 890       |

So far I have been unable to learn a q value function that can be unrolled as a better policy than
boilergoose. If I cannot do it then I could not do either the cycle of evaluation and improvement.

![evaluation and improvement iterative cycle](res/2021-03-15-11-23-08.png)

So I have to solve this problem. Maybe I need a model with more capacity.

| agent  | model | elo score | validation loss |
|--------|-------|-----------|-----------------|
| combo1 | x16   | 890       | 0.054           |
| combo1 | x32   | 1007      | 0.047           |
| combo1 | x64   | 1026      | 0.041           |
| combo1 | x128  | 1026      | 0.040           |

We were able to improve the score, but it is far from the 1269 score of the boilergoose model.

#### Improving RAM memory usage

| experiment                                   | RAM usage (%) | epoch time (s) | GPU usage(%) |
|----------------------------------------------|---------------|----------------|--------------|
| numpy array                                  | 62.26         | 11.2           | 43           |
| generators                                   | 36            | 32             | 16           |
| generator enqueuer                           | 37.9          | 25.8           | 21           |
| generator enqueuer 2 workers multiprocessing | 36.9          | 76             | 8            |
| dataset from generator                       | 36.6          | 26             | 25           |

I have tried to solve the issue of using double RAM memory than expected. However in all the experiments
that I have been able to reduce the RAM usage the epoch size has increased also a lot.

The difference in speed comes because when using numpy arrays it takes a lot of time to start the
training. So it seems to be optimizing something that it does not optimize otherwise. Maybe I need
to solve the issue with ptxas.

I have been able to solve ptxas problem, thus I want to repeat the experiments to see if anything
changes. I will be using a smaller file so I can train with both gpus so I speedup training.
Automate steps per epoch, and create a script for each experiment.

Both dataset from generator and generator enqueuer take twice the time as giving the numpy array as
input.

Using dataset from numpy array is very slow, the speed test reveals that looping over the epoch
takes 13 seconds.

| experiment             | RAM usage (%) | epoch time (s) | GPU usage(%) |
|------------------------|---------------|----------------|--------------|
| numpy array            | 33            | 4.7            | 44           |
| generator enqueuer     | 20            | 11.1           | 20           |
| dataset from generator | 20            | 10.6           | 22           |
| dataset from numpy     | 33            | 16             | 18           |

Some links:

- https://stackoverflow.com/questions/51541610/why-is-tensorflows-tf-data-package-slowing-down-my-code
- https://github.com/tensorflow/tensorflow/issues/39750
- https://github.com/tensorflow/tensorflow/issues/40634
- https://github.com/tensorflow/tensorflow/issues/31312

I think that the best option is to try compiling tensorflow from source. I'm going to do it on
a clone of the conda environment. However I have found that there is a file missing cuda.h that is
necessary to compile tensorflow. So instead I'm going to try to isntall tensorflow from conda.

```bash
conda create -n goose_dev pytest rope pylint tqdm numpy pandas scikit-learn ipython ipykernel tensorflow-gpu=2.4.1
source activate goose_dev
pip install kaggle_environments nvidia-ml-py3 pyyaml psutil
python setup.py develop
conda install -c conda-forge cudatoolkit-dev
```

The disadvantage is that it does not come with cuda11, so it takes forever to start training.

Summary:

- Using numpy and cuda11 is the fastest training, but requires twice as RAM
- Using a generator results in correct RAM usage, but training is twice slower
- Using tensorflow from conda uses cuda10 and it takes forever to start, and uses twice as RAM
- I could not compile tensorflow from source because I don't have a proper cuda installation.
This might the the only viable option.

So I will keep using numpy implementation althought it uses too much RAM. If some day I feel adventurous
I will install cuda and try tensorflow compilation. Maybe installing cuda this days is easier since
I have seen that there is an option to install it with apt.

At least we have installed ptxas and model compilation now is faster.

#### Pretrain on epsilon greedy agent that plays versus greedy agents

I have realized that maybe the approach that I followed when training the random agent does not
have sense anymore. The agent that plays should be epsilon greedy to explore the space, but the
other players should play as good as possible. Otherwise I'm only playing against weak agents.

On this experiment I want to play with the model capacity to see if I can improve results and
also with batch size to see if I can speed up training.

All models have been trained with `ranking_reward_-2_3`.

| experiment         | parameters | loss   | elo score |
|--------------------|------------|--------|-----------|
| greedy baseline    |            | -      | 981       |
| greedy vs greedy   |            | -      | 1083      |
| epsilon vs epsilon |            | -      | 1095      |
| epsilon vs greedy  | x16        | 0.1149 | 1127      |
| epsilon vs greedy  | x32        | 0.1086 | 1080      |
| epsilon vs greedy  | x64        | 0.1042 | 1047      |
| epsilon vs greedy  | x128       | 0.1029 | 1077      |

There does not seem to be a relation between loss and elo scores. Elo scores are similar to previous
trainings. In this case there is no improvement for playing against greedy agents. However we are
already improving over the greedy agent baseline. So probably makes more sense to test this against
better agents.

#### Pretrain on epsilon boilergoose agent that plays versus boilergoose agents

I have not been able to improve over boilergoose, in fact scores are worse than playing against
greedy agent.

| experiment           | reward              | parameters | elo score |
|----------------------|---------------------|------------|-----------|
| boilergoose baseline |                     |            | 1269      |
| epsilon vs greedy    | ranking_reward_-2_3 | x16        | 939       |
| epsilon vs greedy    | ranking_reward_-2_3 | x32        | 1001      |
| epsilon vs greedy    | ranking_reward_-2_3 | x64        | 965       |
| epsilon vs greedy    | ranking_reward_-2_3 | x128       | 986       |
| epsilon vs greedy    | ranking_reward_-4_4 | x16        | 990       |
| epsilon vs greedy    | ranking_reward_-4_4 | x32        | 990       |

Ideas to solve the problem:

- Dig deeper into reward, to do so decouple the game playing of the state generation
- Verify that the state generation is correct, maybe there is an error that
is not hurting too much greedy and random but does hurt boilergoose
- Data augmentation on training with a generator could be useful
- Try with other agents, maybe boilergoose is special

#### Pretrain on epsilon risk_averse_goose agent that plays versus risk_averse_goose agents

### Results

<!---
## Iteration n. title

### Goal

### Development

### Results
--->
