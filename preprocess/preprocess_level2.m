clear variables
clear global
clc
close all
warning off

src_ = '/home/kalvik/shared/CSI_DATA/preprocessed_level1/';
dest_dir = '/home/kalvik/shared/CSI_DATA/preprocessed_level2/';

files = dir(src_);
% remove `.` and `..`
files(1:2) = [];
% select just directories not files
dirFlags = [files.isdir];
% select name of directories
name = files(dirFlags);
name = {name.name};

fs = 1/0.0005;
param_mod = 2;

parfor nm = 1:length(name)
    src_dir = char(strcat(src_, name{nm}, '/'));
    dinfo = dir(char(strcat(src_dir, '*.csv')));

    for K = 1 : length(dinfo)
        thisfilename = dinfo(K).name; % current file

        combined_data = csvread(char(strcat(src_dir, thisfilename))); %load just this file

        amp_data = combined_data(:, 1:270);
        ph_data = combined_data(:, 270:end);

        amp = [];
        ph = [];
        for col = 1:270
            wavelet_amp = wden(amp_data(:, col),'modwtsqtwolog','s','mln',param_mod,'db2');
            wavelet_ph = wden(ph_data(:, col),'modwtsqtwolog','s','mln',param_mod,'db2');

            amp = [amp; wavelet_amp];
            ph = [ph; wavelet_ph];
        end
        amp = transpose(amp);
        ph = transpose(ph);

        combined = [amp, ph];

        % data flush
        folder_name = char(strcat(dest_dir, '/', name{nm}));
        mkdir(folder_name)

        folder_name = char(strcat(folder_name, '/', name{nm}));
        csvwrite([folder_name, num2str(K),'.csv'], combined);
    end
end
