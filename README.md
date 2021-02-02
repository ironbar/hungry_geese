# hungry_geese

![hungry_geese](https://storage.googleapis.com/kaggle-competitions/kaggle/25401/logos/header.png?t=2021-01-22-07-12-33)

Multiplayer agent competition in a Snake like environment

https://www.kaggle.com/c/hungry-geese

## Goal

I want to learn more about MuZero by applying it to this challenge. I do not want to devote too much
time to the challenge, ideally two weeks, less than one month.
I'm planning to present the learnings to the Jedi Council after the challenge.

## Methodology

I'm following [CRISP-DM 1.0](https://www.the-modeling-agency.com/crisp-dm.pdf) methodology for the reports.

I have skipped Evaluation and Deployment steps because they are not usually done on Kaggle.

1. [Business understanding](reports/01_Business_Understanding.md)
1. [Data understanding](reports/02_Data_Understanding.md)
1. [Data preparation](reports/03_Data_Preparation.md)
1. [Modeling](reports/04_Modeling.md)
1. [Solution summary](reports/05_Solution_Summary.md)
* [Challenge workflow](reports/00_Challenge_Workflow.md)

## Code structure

     |_ source
     |_ forum: all the scritps and notebooks taken from the forum
     |_ logs: folder for storing all kind of stats and logs. For example the
     score of each model, results from experiments
     |_ notebooks: jupyter notebooks made during the challenge. They start by number for easier sorting.
     |_ reports: documents made during the challenge according to CRISP-DM methodology
     |_ tests: folder with tests for the library
     |_ data: folder with light data from teh challenge
     |_ rules: the official rules of the challenge