clear; clc;

% BLUEBERRY RIPENESS PREDICTION - Single Image
% Prediksi Kematangan Blueberry dari Satu Gambar

modelPath = 'trainedKNN_Blueberry.mat';
if ~isfile(modelPath)
    error('Model KNN untuk klasifikasi blueberry tidak ditemukan: %s\nJalankan training.m terlebih dahulu.', modelPath);
end
load(modelPath, 'mdl');

imagePath = 'blueberry.png';  % Ganti dengan nama file gambar blueberry Anda
if ~isfile(imagePath)
    error('File gambar tidak ditemukan: %s\nPastikan file berada di direktori kerja yang sama.', imagePath);
end

fprintf('========================================\n');
fprintf('PREDIKSI KEMATANGAN BLUEBERRY\n');
fprintf('========================================\n\n');

Iorig = imread(imagePath);
fprintf('Gambar dimuat: %s\n', imagePath);

Iproc = preprocessImage(Iorig);

hogFeat = extractHOGFeatures(Iproc);
hogFeat = single(hogFeat);

[predictedLabel, scores] = predict(mdl, hogFeat);

fprintf('\nHASIL PREDIKSI:\n');
fprintf('Tingkat Kematangan: %s\n', string(predictedLabel));
fprintf('Confidence: %.2f%%\n\n', max(scores)*100);

fprintf('Perincian Semua Kelas:\n');
[sortedScores, sortedIdx] = sort(scores, 'descend');
classLabels = mdl.ClassNames;
for i = 1:length(classLabels)
    fprintf('  %s: %.2f%%\n', string(classLabels(sortedIdx(i))), sortedScores(i)*100);
end

function Iout = preprocessImage(I)
    % Preprocessing: resize ke 28x28, grayscale, normalize
    I = imresize(I, [28 28]);
    if size(I, 3) == 3
        I = rgb2gray(I);
    end
    I = im2double(I);
    Iout = I;
end