clc
clear all

addpath('/users/kjakkala/neuralwave/preprocess/preprocess_level1/')

srcFolder = '/users/kjakkala/neuralwave/data/WALL';
destFolder = '/users/kjakkala/neuralwave/data/WALL_l1';

d = dir(srcFolder);
isub = [d(:).isdir];
classes = {d(isub).name}';
classes(ismember(classes,{'.','..'})) = [];

dataset = {}
for i = 1:length(classes)
    d = dir(char(strcat(srcFolder, '/', classes(i))));
    isub = ~([d(:).isdir]);
    files = {d(isub).name};

    for j = 1:length(files)
        file = char(strcat(srcFolder, '/', classes(i), '/', files(j)));
        csi_trace = read_csi(file);

	dataset{end+1} = {csi_trace, file}
	hdf5write('/users/kjakkala/myfile.h5', 'dataset', dataset)
%        mkdir(char(strcat(destFolder, '/', classes(i))));
%        file_name = char(files(j));
%	complete_file_name = char(fullfile(destFolder, classes(i), strcat(file_name(1:end-4), '.mat')));
%        save (complete_file_name , 'csi_trace')
    end
end
