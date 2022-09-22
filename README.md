##  Database Reconstruction Attacks (DRA)

> 1️⃣ version: 0.1.0

> ✍️ author: Mitchell Lisle


This notebook is adapted from a paper on database reconstruction attacks. You can find the paper [here.](
https://cacm.acm.org/magazines/2019/3/234925-understanding-database-reconstruction-attacks-on-public-data/fulltext)

There are a number of reasons businesses and governments want to share information about people.
It is important that when sharing information, you consider privacy and anonymity of the people that data is
derived from. In most cases, aggregate data does little to hinder hackers from being able to re-create a database
that is either very close, or exactly the same as the original data. In this application,
we will take a simple example and re-create the database from nothing but aggregate statistics about those people.

We will mainly use [Z3](https://github.com/Z3Prover/z3) for this.


Imagine we have the following database that contains information for people within a certain geographic area
(going forward we refer to this area as a **block**.

We have *7* people in total in this block. Alongside <span style="color:#016FB9">**age**</span>, we also have each resident's
<span style="color:#FF9505">**smoking status**</span>, <span style="color:#DA3E52">**employment status**</span> and
whether they are <span style="color:#6CC551">**married**</span> or not, we publish a variety of
statistics. You have probably seen something similar in your countries census.

> 📓 To simplify the example, this fictional world has:
 - Two <span style="color:#6CC551">**marriage**</span> statuses; Married (**True**) or Single (**False**)
 - Two <span style="color:#FF9505">**smoking**</span> statuses; Non-Smoker (**False**) or Smoker (**True**)
 - Two <span style="color:#DA3E52">**employment statuses**</span>;  Unemployed (**False**) or Employed (**True**)

> 👾 One additional piece of logic we know is that any statistics with a count of less than 3 is suppressed

## Usage
To run 

``shell
dra data/block-stats.csv data/solution.csv data/reconstruction.csv
``