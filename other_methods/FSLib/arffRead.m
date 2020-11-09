function [data, labels] = arffRead(path)
%  function [data, out] = arffRead(path)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
fileID = fopen(path);

%  tline = fgetl(fileID);
%  tline = fgetl(fileID);

% read attributes
fields = {};
ftypes = [];


floop = 1;
fn = 1;

while floop
    tline = fgetl(fileID);
    
    if ~ischar(tline)
        break;
    end
    
    % avoid parsing @DATA and skip blank lines
    if length(tline) > 5 && tline(1) == '@' && strcmpi(tline(2:10),'ATTRIBUTE')
        
        %at = strfind(tline, ' ');
        %
        %if length(at) < 2
        %    error('MATLAB:file','ARFF file not recognized!');
        %end
        %
        %fields{fn} = tline(at(1)+1:at(2)-1);
        %typedef = tline(at(2)+1:end);
        
        % parsing using textscan? (good for data, less for attributes)
        % A = textscan(tline,'%s %s %s','Whitespace',' \t\b{},');
        A = textscan(tline,'%s %s %s');
        
        if isempty(A{1}) || isempty(A{2}) || isempty(A{3})
            fclose(fileID);
            error('MATLAB:file','ARFF file not recognized!');
        end
        
        if size(A{1},1) == 1
            fields{fn} = char(A{2});
            typedef = char(A{3});
        else
            fields{fn} = char(A{2}(1));
            bt = strfind(tline,'{');
            typedef = tline(bt(1):end);
        end
        
        if typedef(1) == '{' && typedef(end) == '}'
            ftypes(fn) = 1;
            %nomspec.(fields{fn}) = typedef;
            
            % out is a cell with parsed classes assuming { x, x, x } format
            out = textscan(typedef, '%s', 'Delimiter', ' ,{}', 'MultipleDelimsAsOne', 1);
            % nclasses = length(out{1});
                        
           
            % expand cell (avoid cell of cell)
            nomspec.(fields{fn}) = out{:};
        else
            if strcmpi(typedef,'NUMERIC')
                ftypes(fn) = 0;
            elseif strcmpi(typedef,'STRING')
                ftypes(fn) = 2;
            elseif strcmpi(typedef,'REAL')
                ftypes(fn) = 0;                
            else
                dt = strfind(typedef, ' ');
                
                if ~isempty(dt) && strcmpi(typedef(1:dt(1)-1), 'DATE')
                    ftypes(fn) = 3;
                    % implement date-format parsing
                else
                    fclose(fileID);
                    error('MATLAB:file','ARFF file not recognized!');
                end
            end
        end
        
        fn = fn + 1;
    end
end

%  disp(numel(ftypes))

numerColumn = ones(1,numel(ftypes)-1);
numerColumn = [numerColumn 0];
txtSpec = '';
% generate parser
for i = 1:length(ftypes)
    if ftypes(i)==0
        txtSpec = strcat(txtSpec,' %f');
        % numerColumn(i) = 1;
    else
        samp = nomspec.(fields{i});
        while iscell(samp)
            samp = samp{1};
        end
        if isnumeric(samp) || ~isnan(str2double(samp))
            txtSpec = strcat(txtSpec,' %f');
            % numerColumn(i) = 1;
        else
            txtSpec = strcat(txtSpec,' %s');
        end
    end
end

txtSpec(1) = [];

% rewind file
fseek(fileID,0,-1);

% seek data
has_data = 0;

while floop
    tline = fgetl(fileID);
    
    if length(tline) == 5 && strcmpi(tline(1:5),'@DATA')
        has_data = 1;
        break;
    end
    
    if ~ischar(tline)
        break;
    end
end

data = textscan(fileID, txtSpec, 'Delimiter',',');

N = length(data{:,end});

labels = zeros(N,1);

tipocell = 0;
if strcmp(class(data{end}(1)),'cell'),
    tipocell = 1;
end;    

for i=1:N,
      if (tipocell),
           labels(i) = find(strcmp(out{1}, data{end}(i)));
      else
           labels(i) = find(strcmp(out{1}, num2str(data{end}(i)) ));
      end;
end;      
data = cell2mat(data(logical(numerColumn)));

end

