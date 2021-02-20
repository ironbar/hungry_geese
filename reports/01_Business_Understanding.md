# Business Understanding
<!--- --->
## Challenge description

<!--- Look at the challenge description, understand the goal of the challenge
and write it here with your own words. Use images if they improve the explanation--->

> You will create an AI agent to play against others and survive the longest. You must make sure
your goose doesn’t starve or run into other geese; it’s a good thing that geese love peppers,
donuts, and pizza—which show up across the board.

## Evaluation

<!--- Understand the metric used on the challenge, write it here and study
the characteristics of the metric --->

> Each day, your team is able to submit up to 5 agents (bots) to the competition. Each submission will play episodes (games) against other bots on the ladder that have a similar skill rating. Over time, skill ratings will go up with wins or down with losses.

<!------>

> Each Submission has an estimated Skill Rating which is modeled by a Gaussian N(μ,σ2) where μ is the estimated skill and σ represents our uncertainty of that estimate which will decrease over time.

<!------>

> At the submission deadline, additional submissions will be locked. One additional week will be allotted to continue to run games. At the conclusion of this week, the leaderboard is final.

I would like to see code for the ranking algorithm, it is briefly explained here but I would like
to know the exact implementation.

> After an Episode finishes, we'll update the Rating estimate of both agents in that Episode. If one agent won, we'll increase its μ and decrease its opponent's μ -- if the result was a draw, then we'll move the two μ values closer towards their mean. The updates will have magnitude relative to the deviation from the expected result based on the previous μ values and also relative to each Submission's uncertainty σ. We also reduce the σ terms relative to the amount of information gained by the result. The score by which your bot wins or loses an Episode does not affect the skill rating updates.

## Assess situation

<!---This task involves more detailed fact-finding about all of the resources,
constraints, assumptions, and other factors that should be considered in determining
the data analysis goal and project plan

- timeline. Is there any week where I could not work on the challenge?
- resources. Is there any other project competing for resources?
- other projects. May I have other more interesting projects in the horizon?
 --->

I don't want to devote too much time to this project: ideally two weeks, 1 month maximun. Thus I should
have done all the work by 7 March 2021.

I will have my new pc for developing and also the notebooks to measure the execution time.

### Terminology

<!--- Sometimes the field of the challenge has specific terms, if that is the
case write them here, otherwise delete this section.--->

## Project Plan

<!--- Write initial ideas for the project. This is just initial thoughts,
during the challenge I will have a better understanding of the project and
with better information I could decide other actions not considered here.--->

The initial plan was to implement MuZero algorithm for this challenge. I think it is a very interesting algorithm
because it uses a world model and does plan with it.

However after learning more about the challenge I have decided to broaden the scope. I will start
with something very simple like Q-learning and mantain a league of agents similar to AlphaStar one.
This will help me to learn more about Reinforcement Learning. I could start with MuZero but I don't
think it has too much sense.
