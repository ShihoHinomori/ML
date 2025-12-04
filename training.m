clear; clc;

% BLUEBERRY RIPENESS CLASSIFICATION - KNN Training Pipeline
% Klasifikasi Kematangan Buah Blueberry menggunakan KNN

% ===== ROBUST PATH RESOLUTION =====
currentFile = mfilename('fullpath');
projectDir = fileparts(currentFile);
rootFolder = fullfile(projectDir, 'sample');

defaultK = 5;  % Number of neighbors

% ===== VALIDASI FOLDER =====
if ~exist(rootFolder, 'dir')
    fprintf('\n❌ ERROR: Folder dataset tidak ditemukan!\n');
    fprintf('Path yang dicari: %s\n\n', rootFolder);
    fprintf('Solusi: Buat struktur folder berikut:\n');
    fprintf('  %s\\sample\\immature\\\n', projectDir);
    fprintf('  %s\\sample\\semi-mature\\\n', projectDir);
    fprintf('  %s\\sample\\mature\\\n\n', projectDir);
    error('Folder dataset tidak ditemukan: %s', rootFolder);
end

imds = imageDatastore(rootFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

if isempty(imds.Files)
    error('Tidak ada file gambar ditemukan di folder: %s', rootFolder);
end

numImages = numel(imds.Files);
fprintf('\n========================================\n');
fprintf('KLASIFIKASI KEMATANGAN BLUEBERRY - KNN\n');
fprintf('========================================\n\n');
fprintf('=== INFORMASI DATASET ===\n');
fprintf('Total gambar blueberry: %d\n', numImages);
fprintf('Distribusi per tingkat kematangan:\n');
labelCounts = countEachLabel(imds);
disp(labelCounts);

fprintf('\n=== PREPROCESSING & EKSTRAKSI FITUR HOG ===\n');
fprintf('Memproses gambar blueberry (resize ke 28x28, konversi grayscale)...\n');
fprintf('Mengekstrak Histogram of Oriented Gradients (HOG)...\n');

I0 = readimage(imds, 1);
I0 = imresize(I0, [28 28]);
if size(I0,3)==3
    I0 = rgb2gray(I0);
end
[hogFeat, ~] = extractHOGFeatures(I0);
hogLength = length(hogFeat);

X = zeros(numImages, hogLength);
Y = cell(numImages,1);

for i = 1:numImages
    I = readimage(imds, i);
    
    I = imresize(I, [28 28]);
    if size(I,3)==3
        I = rgb2gray(I);
    end
    
    X(i,:) = extractHOGFeatures(I);
    Y{i} = char(imds.Labels(i));
    
    if mod(i, 1000) == 0
        fprintf('Processed: %d/%d\n', i, numImages);
    end
end
Y = categorical(Y);

uniqueLabels = categories(Y);
numClasses = length(uniqueLabels);

Xtrain = X;
Ytrain = Y;
Xtest = X;
Ytest = Y;

fprintf('Using all %d samples for both training and testing\n', numImages);

fprintf('\n=== PELATIHAN MODEL KNN ===\n');
k = defaultK;
fprintf('Melatih K-Nearest Neighbors (KNN) dengan k=%d neighbors...\n', k);

tic;
mdl = fitcknn(Xtrain, Ytrain, ...
              'NumNeighbors', k, ...
              'Distance', 'euclidean', ...
              'Standardize', true);
save('trainedKNN_Blueberry.mat', 'mdl');
fprintf('✓ Model KNN berhasil disimpan ke "trainedKNN_Blueberry.mat"\n');

trainTime = toc;

tic;
Ypred = predict(mdl, Xtest);
predTime = toc;

[acc, precision, recall, f1, support] = evaluateModelDetailed(Ytest, Ypred, uniqueLabels);

fprintf('\n=== HASIL PERFORMA MODEL ===\n');
fprintf('Waktu Training: %.4f detik\n', trainTime);
fprintf('Waktu Prediksi: %.4f detik\n', predTime);
fprintf('\n');

displayClassificationReport(uniqueLabels, precision, recall, f1, support, acc);

figure('Position', [100, 100, 800, 600]);
confusionchart(Ytest, Ypred);
title(sprintf('Confusion Matrix - Klasifikasi Kematangan Blueberry (k=%d, Akurasi=%.2f%%)', k, acc*100));

function [acc, precision, recall, f1, support] = evaluateModelDetailed(Ytest, Ypred, uniqueLabels)
    acc = mean(Ypred == Ytest);
    C = confusionmat(Ytest, Ypred);
    
    numClasses = length(uniqueLabels);
    if size(C,1) < numClasses || size(C,2) < numClasses
        newC = zeros(numClasses, numClasses);
        predictedLabels = categories(Ypred);
        trueLabels = categories(Ytest);
        
        for i = 1:length(trueLabels)
            trueIdx = find(strcmp(uniqueLabels, trueLabels{i}));
            for j = 1:length(predictedLabels)
                predIdx = find(strcmp(uniqueLabels, predictedLabels{j}));
                if ~isempty(trueIdx) && ~isempty(predIdx)
                    newC(trueIdx, predIdx) = C(i, j);
                end
            end
        end
        C = newC;
    end
    
    precision = diag(C) ./ sum(C, 2);
    recall = diag(C) ./ sum(C, 1)';
    support = sum(C, 2);
    
    precision(isnan(precision)) = 0;
    recall(isnan(recall)) = 0;
    
    f1 = 2 * (precision .* recall) ./ (precision + recall);
    f1(isnan(f1)) = 0;
end

function displayClassificationReport(labels, precision, recall, f1, support, accuracy)
    fprintf('Classification Report - Klasifikasi Kematangan Blueberry:\n\n');
    fprintf('%12s %9s %8s %9s %9s\n', '', 'precision', 'recall', 'f1-score', 'support');
    fprintf('\n');
    
    for i = 1:length(labels)
        fprintf('%12s %9.6f %8.6f %9.6f %9d\n', ...
            char(labels(i)), precision(i), recall(i), f1(i), support(i));
    end
    
    fprintf('\n');
    
    totalSupport = sum(support);
    macroAvgPrecision = mean(precision);
    macroAvgRecall = mean(recall);
    macroAvgF1 = mean(f1);
    
    weightedAvgPrecision = sum(precision .* support) / totalSupport;
    weightedAvgRecall = sum(recall .* support) / totalSupport;
    weightedAvgF1 = sum(f1 .* support) / totalSupport;
    
    fprintf('%12s %9s %8s %9.6f %9d\n', 'accuracy', '', '', accuracy, totalSupport);
    fprintf('%12s %9.6f %8.6f %9.6f %9d\n', 'macro avg', macroAvgPrecision, macroAvgRecall, macroAvgF1, totalSupport);
    fprintf('%12s %9.6f %8.6f %9.6f %9d\n', 'weighted avg', weightedAvgPrecision, weightedAvgRecall, weightedAvgF1, totalSupport);
end