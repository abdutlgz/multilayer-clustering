"""MovieLens 25M: users, genres and layers (Section 5.1 of the paper).

A layer is a binary users x movies matrix: entry (i, j) is 1 if user i rated movie j.
All layers have the same n = 5,000 users (the users with the most ratings).
For each of the 8 genres, a random 75% of its movies are divided into 4 equal large layers and the
other 25% into S equal small layers. A movie with several genres appears in layers of each of them.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

N_USERS = 5000
MIN_RATINGS = 40          # users are taken from those with at least 40 ratings
GENRES = ['Action', 'Adventure', 'Comedy', 'Crime', 'Drama', 'Romance', 'Sci-Fi', 'Thriller']
LARGE_PER_GENRE = 4
LARGE_SHARE = 0.75


def data_dir():
    """Folder with ratings.csv and movies.csv: $ML25M_DIR, or realworldexample/data/ml-25m."""
    default = Path(__file__).resolve().parents[1] / 'data' / 'ml-25m'
    return Path(os.environ.get('ML25M_DIR', default))


class MovieLens:
    """The 5,000 users with the most ratings, and the movies of each genre that they rated."""

    def __init__(self, path=None):
        path = Path(path) if path else data_dir()
        ratings = pd.read_csv(path / 'ratings.csv', usecols=['userId', 'movieId'], dtype=np.int64)
        movies = pd.read_csv(path / 'movies.csv')

        counts = ratings.userId.value_counts()
        counts = counts[counts >= MIN_RATINGS]
        users = np.sort(counts.sort_values(ascending=False, kind='stable').index[:N_USERS].values)

        r = ratings[ratings.userId.isin(users)].copy()
        r['row'] = pd.Series(np.arange(N_USERS), index=users)[r.userId].values
        rated = set(r.movieId.unique())
        genre_lists = movies.genres.str.split('|')

        self.n = N_USERS
        # movie ids of each genre (rated by at least one of the users), sorted
        self.genre_movies = {
            g: np.array(sorted(m for m in movies.movieId[genre_lists.apply(lambda x: g in x)] if m in rated))
            for g in GENRES}
        # rows (users) who rated each movie
        self.raters = r.groupby('movieId').row.apply(np.array).to_dict()

    def layer(self, movie_ids):
        """Binary n x len(movie_ids) matrix: (i, j) = 1 if user i rated movie_ids[j]."""
        rows = [self.raters[m] for m in movie_ids]
        cols = [np.full(len(x), j) for j, x in enumerate(rows)]
        return sparse.csc_matrix((np.ones(sum(map(len, rows))), (np.concatenate(rows), np.concatenate(cols))),
                                 shape=(self.n, len(movie_ids)))

    def divide(self, n_small, rep):
        """Layers of one random division of the movies, with n_small small layers per genre.

        Returns a list of dicts {'genre', 'type' ('large' or 'small'), 'a' (sparse matrix)}, ordered by genre;
        within a genre, the 4 large layers come first. Division `rep` of a given n_small is always the same.
        """
        rng = np.random.default_rng([20261108, 7, n_small, rep])
        layers = []
        for g in GENRES:
            m = rng.permutation(self.genre_movies[g])
            n_large = int(round(LARGE_SHARE * len(m)))
            for p in np.array_split(m[:n_large], LARGE_PER_GENRE):
                layers.append({'genre': g, 'type': 'large', 'a': self.layer(p)})
            for p in np.array_split(m[n_large:], n_small):
                layers.append({'genre': g, 'type': 'small', 'a': self.layer(p)})
        return layers
