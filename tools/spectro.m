data=csvread('/home/kjakkala/neuralwave/data/random_data_l1/ting_li/ting1.csv');

data=data(:,:);
[s,f,t]=spectrogram(data(:,1),gausswin(1024),986,1024,1000,'yaxis');

x=zeros(80,length(t));
%for k=1:5
for k=1:20
    [s,f,t]=spectrogram(data(:,k),gausswin(1024),986,1024,1000,'yaxis');
    s_magnitude=abs(s);
    s_first80=s_magnitude(1:80,:);
    s_enery_level=sum(s_first80);
    a_magnitude_norm=s_first80./s_enery_level;
    s_magnitude_mean=mean(a_magnitude_norm,2);
    s_magnitude_denoise=a_magnitude_norm-s_magnitude_mean;
    index=s_magnitude_denoise<0;
    s_magnitude_denoise(index)=0;
    x=x+s_magnitude_denoise;
end 
person3=x(:, :);
imagesc(flipud(person3));
xticklabels = 0:1:10;
xticks = linspace(0, size(person3, 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels,'fontsize', 15)
xlabel('time(s)','fontsize',15)
yticklabels = 0:20:99.6;
yticks = linspace(0, size(person3, 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel',flip(yticklabels),'fontsize', 15)
ylabel('frequency(Hz)','fontsize',15)