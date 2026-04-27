function [A_layers, layer_info, selectedUserIDs] = build_movielens_genre_splits( ...
    ratingsTbl, moviesTbl, ...
    minUserRatings, maxNumUsers, ...
    selectedGenres, numSplits, splitMode, rngSeed, ...
    usePositiveOnly, positiveThreshold)

A_layers = {};
layer_info = table();
selectedUserIDs = [];

rng(rngSeed);

%% Step 1: filter users by activity
[userGroups, userIDs_unique] = findgroups(ratingsTbl.userId);
userCountVals = splitapply(@numel, ratingsTbl.movieId, userGroups);

keepMask = userCountVals >= minUserRatings;
activeUsers = userIDs_unique(keepMask);
activeCounts = userCountVals(keepMask);

if isempty(activeUsers)
    error('No users remain after filtering by minUserRatings.');
end

[~, ord] = sort(activeCounts, 'descend');
activeUsers = activeUsers(ord);
activeUsers = activeUsers(1:min(maxNumUsers, numel(activeUsers)));

ratingsTbl = ratingsTbl(ismember(ratingsTbl.userId, activeUsers), :);

if usePositiveOnly
    ratingsTbl = ratingsTbl(ratingsTbl.rating >= positiveThreshold, :);
end

if isempty(ratingsTbl)
    error('No ratings remain after filtering.');
end

%% Step 2: row mapping
selectedUserIDs = unique(ratingsTbl.userId);
selectedUserIDs = sort(selectedUserIDs);
n = numel(selectedUserIDs);

userRowMap = containers.Map('KeyType', 'double', 'ValueType', 'double');
for i = 1:n
    userRowMap(selectedUserIDs(i)) = i;
end

%% Step 3: parse movie genres
movieGenreLists = cell(height(moviesTbl),1);
for i = 1:height(moviesTbl)
    g = moviesTbl.genres{i};
    if strcmp(g, '(no genres listed)')
        movieGenreLists{i} = {};
    else
        movieGenreLists{i} = strsplit(g, '|');
    end
end

%% Step 4: build layers = genre x split
A_layers_cell = {};
layer_id_list = [];
genre_list = {};
split_list = [];
n_users_list = [];
n_movies_list = [];
n_edges_list = [];
beta_list = [];
movie_ids_list = {};

layerCounter = 0;

for g = 1:numel(selectedGenres)
    thisGenre = selectedGenres{g};

    isGenreMovie = false(height(moviesTbl),1);
    for i = 1:height(moviesTbl)
        isGenreMovie(i) = any(strcmp(movieGenreLists{i}, thisGenre));
    end

    genreMovieIDs = moviesTbl.movieId(isGenreMovie);

    % keep only movies actually observed in ratings
    genreRatings = ratingsTbl(ismember(ratingsTbl.movieId, genreMovieIDs), :);
    if isempty(genreRatings)
        continue;
    end

    layerMovieIDs = unique(genreRatings.movieId);
    layerMovieIDs = sort(layerMovieIDs);

    if numel(layerMovieIDs) < numSplits
        warning('Genre %s has fewer movies than numSplits; skipping.', thisGenre);
        continue;
    end

    % randomize movie order
    perm = randperm(numel(layerMovieIDs));
    layerMovieIDs = layerMovieIDs(perm);

    % split edges
    chunkEdges = make_split_edges(numel(layerMovieIDs), numSplits, splitMode);

    for s = 1:numSplits
        idx1 = chunkEdges(s) + 1;
        idx2 = chunkEdges(s+1);

        if idx1 > idx2
            continue;
        end

        subMovies = layerMovieIDs(idx1:idx2);
        n_l = numel(subMovies);

        if n_l < 2
            continue;
        end

        layerRatings = genreRatings(ismember(genreRatings.movieId, subMovies), :);
        if isempty(layerRatings)
            continue;
        end

        movieColMap = containers.Map('KeyType', 'double', 'ValueType', 'double');
        for j = 1:n_l
            movieColMap(subMovies(j)) = j;
        end

        pairs = unique([layerRatings.userId, layerRatings.movieId], 'rows');

        numPairs = size(pairs,1);
        rowIdx = zeros(numPairs,1);
        colIdx = zeros(numPairs,1);
        keep = false(numPairs,1);

        for k = 1:numPairs
            uid = pairs(k,1);
            mid = pairs(k,2);

            if isKey(userRowMap, uid) && isKey(movieColMap, mid)
                rowIdx(k) = userRowMap(uid);
                colIdx(k) = movieColMap(mid);
                keep(k) = true;
            end
        end

        rowIdx = rowIdx(keep);
        colIdx = colIdx(keep);

        if isempty(rowIdx)
            continue;
        end

        A_l = sparse(rowIdx, colIdx, 1, n, n_l);

        if nnz(A_l) == 0
            continue;
        end

        layerCounter = layerCounter + 1;
        A_layers_cell{layerCounter,1} = A_l;

        layer_id_list(layerCounter,1) = layerCounter;
        genre_list{layerCounter,1} = thisGenre;
        split_list(layerCounter,1) = s;
        n_users_list(layerCounter,1) = n;
        n_movies_list(layerCounter,1) = n_l;
        n_edges_list(layerCounter,1) = nnz(A_l);
        beta_list(layerCounter,1) = log(n_l) / log(n);
        movie_ids_list{layerCounter,1} = subMovies;
    end
end

if isempty(A_layers_cell)
    error('No layers were created.');
end

A_layers = A_layers_cell;

layer_info = table( ...
    layer_id_list, genre_list, split_list, ...
    n_users_list, n_movies_list, n_edges_list, beta_list, movie_ids_list, ...
    'VariableNames', {'layer_id','genre','split_id','n_users','n_movies','n_edges','beta_l','movie_ids'});

layer_info = sortrows(layer_info, {'genre','split_id'}, {'ascend','ascend'});
old_ids = layer_info.layer_id;
A_layers = A_layers(old_ids);
layer_info.layer_id = (1:height(layer_info))';
end

function chunkEdges = make_split_edges(N, numSplits, splitMode)
% Returns integer boundaries [0, ..., N]

switch lower(splitMode)
    case 'equal'
        chunkEdges = round(linspace(0, N, numSplits+1));

    case 'uneven'
        % deterministic uneven proportions, then scaled to sum to N
        w = linspace(0.65, 1.35, numSplits);
        w = w(randperm(numSplits)); % shuffle sizes
        w = w / sum(w);

        sizes = floor(N * w);
        remN = N - sum(sizes);

        % distribute remainder
        for t = 1:remN
            sizes(mod(t-1, numSplits) + 1) = sizes(mod(t-1, numSplits) + 1) + 1;
        end

        % avoid empty splits if possible
        if N >= numSplits
            zeroIdx = find(sizes == 0);
            for z = 1:numel(zeroIdx)
                donor = find(sizes > 1, 1, 'first');
                if ~isempty(donor)
                    sizes(donor) = sizes(donor) - 1;
                    sizes(zeroIdx(z)) = 1;
                end
            end
        end

        chunkEdges = [0; cumsum(sizes(:))];
        chunkEdges(end) = N;

    otherwise
        error('Unknown splitMode. Use ''equal'' or ''uneven''.');
end
end