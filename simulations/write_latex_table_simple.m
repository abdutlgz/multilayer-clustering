function write_latex_table_simple(T, outFile)
%WRITE_LATEX_TABLE_SIMPLE Write a dependency-free LaTeX tabular with hlines.

if ~istable(T)
    T = cell2table(T);
end

[outDir,~,~] = fileparts(outFile);
if ~isempty(outDir) && ~exist(outDir, 'dir')
    mkdir(outDir);
end

fid = fopen(outFile, 'w');
if fid < 0
    error('write_latex_table_simple:OpenFailed', 'Could not open %s for writing.', outFile);
end
cleaner = onCleanup(@() fclose(fid));

varNames = T.Properties.VariableNames;
numCols = numel(varNames);

fprintf(fid, '\\begin{tabular}{%s}\n', repmat('l', 1, numCols));
fprintf(fid, '\\hline\n');
for c = 1:numCols
    fprintf(fid, '%s', latex_escape(varNames{c}));
    if c < numCols
        fprintf(fid, ' & ');
    else
        fprintf(fid, ' \\\\\n');
    end
end
fprintf(fid, '\\hline\n');

for r = 1:height(T)
    for c = 1:numCols
        value = T{r,c};
        fprintf(fid, '%s', latex_escape(format_value(value)));
        if c < numCols
            fprintf(fid, ' & ');
        else
            fprintf(fid, ' \\\\\n');
        end
    end
end

fprintf(fid, '\\hline\n');
fprintf(fid, '\\end{tabular}\n');
end

function s = format_value(value)
if iscell(value)
    if isempty(value)
        s = '';
    else
        s = format_value(value{1});
    end
elseif isstring(value)
    if isempty(value)
        s = '';
    else
        s = char(value(1));
    end
elseif ischar(value)
    s = value;
elseif isnumeric(value)
    if isempty(value) || (isscalar(value) && isnan(value))
        s = '';
    elseif isscalar(value)
        if abs(value - round(value)) < 1e-10
            s = sprintf('%d', round(value));
        else
            s = sprintf('%.4g', value);
        end
    else
        s = strjoin(cellstr(string(value(:)')), ', ');
    end
elseif islogical(value)
    s = char(string(value));
else
    s = char(string(value));
end
end

function s = latex_escape(s)
s = char(s);
s = strrep(s, '&', '\&');
s = strrep(s, '%', '\%');
s = strrep(s, '$', '\$');
s = strrep(s, '#', '\#');
s = strrep(s, '_', '\_');
s = strrep(s, '{', '\{');
s = strrep(s, '}', '\}');
end
