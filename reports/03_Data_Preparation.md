# Data Preparation

## Construct Data

<!---This task includes constructive data preparation operations such as the
production of derived attributes or entire new records, or transformed values
for existing attributes. --->

I'm going to enumerate the features that I think we have to feed the model.

| feature                                   | motivation                                                         |
|-------------------------------------------|--------------------------------------------------------------------|
| Time to die (by shrinking)                | The closest I'm to starving the more risk I will take to eat       |
| Time to shrink                            | This could be useful for planning moves                            |
| Difference in size with the other geesse  | If I'm already the bigger goose maybe I don't have to grow         |
| Time to episode end                       | Policy will likely be different at the start or end of the episode |

In the board I should show:

- Heads
- Tails
- Body
- Available future moves

Since all that matters is the ranking I'm going to create a new reward that is exactly the ranking.

| situation         | reward                                                         |
|-------------------|----------------------------------------------------------------|
| agent dies        | -1                                                             |
| other agent dies  | +1                                                             |
| reach final state | +1 for other agent below, +0.5 for other agent with same score |

I could make distinctions if many agents die at the same time, but I don't think it has too much sense.
The agent needs to stay alive.
With this rewards the agent will learn to stay alive, to predict when the other agents are going
to die (and maybe to kill them), and to reach the final state as big as possible.

## Integrate Data
<!---These are methods whereby information is combined from multiple tables or
records to create new recordsor values. --->
## Format Data
<!---Formatting transformations refer to primarily syntactic modifications made
to the data that do not change its meaning, but might be required by the
modeling tool. --->
