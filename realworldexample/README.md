# MovieLens Real-Data Experiment

This folder contains MATLAB code for the MovieLens real-data experiment used in the paper.

The goal of the experiment is to construct a heterogeneous multilayer user--movie network and apply the proposed active-layer clustering and inactive-layer classification pipeline.

## Dataset

We use the MovieLens 25M dataset (`ml-25m`) from GroupLens. The dataset contains 25,000,095 ratings and 1,093,360 tag applications across 62,423 movies from 162,541 users. The data were generated on November 21, 2019. Users are anonymized and no demographic information is included. The dataset files include `ratings.csv`, `movies.csv`, `tags.csv`, `links.csv`, `genome-scores.csv`, and `genome-tags.csv`. :contentReference[oaicite:0]{index=0}

The raw MovieLens data are **not included** in this repository due to the dataset license. Please download the data directly from GroupLens:

```text
https://grouplens.org/datasets/movielens/
