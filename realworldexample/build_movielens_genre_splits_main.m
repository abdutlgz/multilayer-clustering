%% build_movielens_genre_splits_main.m
clear; clc;

% ---------- locate data folder ----------
candidate1 = pwd;
candidate2 = fullfile(pwd, 'ml-25m');
candidate3 = fullfile(pwd, 'ml_25');

if isfile(fullfile(candidate1, 'ratings.csv')) && isfile(fullfile(candidate1, 'movies.csv'))
    dataFolder = candidate1;
elseif isfile(fullfile(candidate2, 'ratings.csv')) && isfile(fullfile(candidate2, 'movies.csv'))
    dataFolder = candidate2;
elseif isfile(fullfile(candidate3, 'ratings.csv')) && isfile(fullfile(candidate3, 'movies.csv'))
    dataFolder = candidate3;
else
    error('Could not find ratings.csv and movies.csv.');
end

fprintf('Loading MovieLens data...\n');
[ratingsTbl, moviesTbl] = load_movielens25_tables(dataFolder);

% ---------- settings ----------
minUserRatings = 40;
maxNumUsers = 8000;

selectedGenres = { ...
    'Action', 'Adventure', 'Comedy', 'Crime', ...
    'Drama', 'Romance', 'Sci-Fi', 'Thriller'};

numSplits = 8;              % 8 genres x 8 splits = up to 64 layers
splitMode = 'uneven';       % 'equal' or 'uneven'
rngSeed = 7;                % reproducibility

usePositiveOnly = false;
positiveThreshold = 4.0;

saveFileName = 'movielens_genre_split_layers.mat';

fprintf('Building genre-split layers...\n');
[A_layers, layer_info, user_ids] = build_movielens_genre_splits( ...
    ratingsTbl, moviesTbl, ...
    minUserRatings, maxNumUsers, ...
    selectedGenres, numSplits, splitMode, rngSeed, ...
    usePositiveOnly, positiveThreshold);

fprintf('\nSummary:\n');
disp(layer_info(:, {'layer_id','genre','split_id','n_users','n_movies','n_edges','beta_l'}));

fprintf('Saving results to %s ...\n', saveFileName);
save(saveFileName, 'A_layers', 'layer_info', 'user_ids', '-v7.3');

fprintf('Done.\n');

fprintf('\nActive-layer counts by threshold:\n');
for b = [0.5 0.6 0.7 0.8 0.90 1.00 1.05 1.10 1.15 1.20]
    fprintf('beta*=%.2f -> active layers = %d\n', b, sum(layer_info.beta_l >= b));
end