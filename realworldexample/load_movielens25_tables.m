function [ratingsTbl, moviesTbl] = load_movielens25_tables(dataFolder)
% Load MovieLens ratings and movies tables

ratingsFile = fullfile(dataFolder, 'ratings.csv');
moviesFile  = fullfile(dataFolder, 'movies.csv');

if ~isfile(ratingsFile)
    error('ratings.csv not found at: %s', ratingsFile);
end
if ~isfile(moviesFile)
    error('movies.csv not found at: %s', moviesFile);
end

ratingsTbl = readtable(ratingsFile);
moviesTbl  = readtable(moviesFile);

requiredRatingsVars = {'userId','movieId','rating','timestamp'};
requiredMoviesVars  = {'movieId','title','genres'};

for i = 1:numel(requiredRatingsVars)
    if ~ismember(requiredRatingsVars{i}, ratingsTbl.Properties.VariableNames)
        error('ratings.csv missing required column: %s', requiredRatingsVars{i});
    end
end

for i = 1:numel(requiredMoviesVars)
    if ~ismember(requiredMoviesVars{i}, moviesTbl.Properties.VariableNames)
        error('movies.csv missing required column: %s', requiredMoviesVars{i});
    end
end

% Convert timestamp to datetime
ratingsTbl.datetime = datetime(ratingsTbl.timestamp, ...
    'ConvertFrom', 'posixtime', 'TimeZone', 'UTC');

end