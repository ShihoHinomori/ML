clear; clc;

% BLUEBERRY RIPENESS PREDICTION - Single Image
% Prediksi Kematangan Blueberry dari Satu Gambar

% ===== SETUP PATH =====
currentFile = mfilename('fullpath');
projectDir = fileparts(currentFile);

% ===== KONFIGURASI =====
% Ganti path ini ke gambar blueberry yang ingin diprediksi
imagePath = fullfile(projectDir, 'sample', 'mature', 'DSCF5288_JPG-rf_roi_350_jpg.rf.e96c512e40085237d84f20b57e61ca33.jpg');
modelPath = fullfile(projectDir, 'trainedKNN_Blueberry.mat');

% ===== VALIDASI FILE =====
if ~isfile(modelPath)
    fprintf('\n❌ ERROR: Model tidak ditemukan!\n');
    fprintf('Path: %s\n\n', modelPath);
    fprintf('Solusi: Jalankan training.m terlebih dahulu\n');
    error('Model file not found: %s', modelPath);
end

if ~isfile(imagePath)
    fprintf('\n❌ ERROR: Gambar tidak ditemukan!\n');
    fprintf('Path: %s\n\n', imagePath);
    fprintf('Tips: Ubah variabel imagePath di predict_single_image.m ke:\n');
    fprintf('  fullfile(projectDir, ''sample'', ''mature'', ''nama_file.jpg'')\n');
    fprintf('atau path absolut seperti:\n');
    fprintf('  ''C:\\Users\\...\\blueberry.jpg''\n');
    error('Image file not found: %s', imagePath);
end

% ===== LOAD MODEL =====
fprintf('\n========================================\n');
fprintf('PREDIKSI KEMATANGAN BLUEBERRY - KNN\n');
fprintf('========================================\n\n');
fprintf('✓ Loading model dari: %s\n', modelPath);
load(modelPath, 'mdl');
fprintf('✓ Model berhasil dimuat\n\n');

% ===== BACA DAN PREPROCESS GAMBAR =====
fprintf('=== PREPROCESSING GAMBAR ===\n');
fprintf('Path gambar: %s\n', imagePath);
I = imread(imagePath);
fprintf('✓ Ukuran asli: %d×%d\n', size(I,1), size(I,2));

% Standardisasi ukuran dan warna
I_proc = imresize(I, [28 28]);
if size(I_proc, 3) == 3
    I_proc = rgb2gray(I_proc);
end
I_proc = im2double(I_proc);
fprintf('✓ Preprocessing: resize 28×28, konversi grayscale\n');

% ===== EKSTRAKSI FITUR HOG =====
fprintf('\n=== EKSTRAKSI FITUR HOG ===\n');
hogFeature = extractHOGFeatures(I_proc);
hogFeature = single(hogFeature);
fprintf('✓ HOG Feature Length: %d\n', length(hogFeature));

% ===== PREDIKSI =====
fprintf('\n=== PREDIKSI MODEL KNN ===\n');
[predictedLabel, scores] = predict(mdl, hogFeature);
fprintf('✓ Prediksi selesai\n\n');

% ===== HASIL PREDIKSI =====
fprintf('========================================\n');
fprintf('HASIL PREDIKSI\n');
fprintf('========================================\n');
fprintf('Tingkat Kematangan: %s\n', char(predictedLabel));
fprintf('Confidence Score: %.2f%%\n\n', max(scores)*100);

fprintf('Perincian Semua Kelas:\n');
[sortedScores, sortedIdx] = sort(scores, 'descend');
classLabels = mdl.ClassNames;
for i = 1:length(classLabels)
    fprintf('  %s: %.2f%%\n', string(classLabels(sortedIdx(i))), sortedScores(i)*100);
end

% ===== VISUALISASI =====
figure('Position', [100, 100, 1000, 400], 'Name', 'Blueberry Ripeness Prediction');

% Gambar asli
subplot(1,3,1);
imshow(I);
title('Gambar Asli', 'FontSize', 12, 'FontWeight', 'bold');
xlabel(sprintf('Ukuran: %d×%d', size(I,1), size(I,2)));

% Gambar setelah preprocessing
subplot(1,3,2);
imshow(I_proc);
title('Preprocessing (28×28)', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Grayscale');

% Hasil prediksi
subplot(1,3,3);
imshow(I_proc);
hold on;
% Tambahkan text dengan background
text(14, -3, sprintf('Prediksi: %s', char(predictedLabel)), ...
    'FontSize', 13, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center', ...
    'BackgroundColor', 'yellow', 'EdgeColor', 'black', 'LineWidth', 1.5);
title('Hasil Prediksi', 'FontSize', 12, 'FontWeight', 'bold');
hold off;

sgtitle(sprintf('Klasifikasi Kematangan Blueberry - Prediksi: %s (%.2f%%)', char(predictedLabel), max(scores)*100), ...
    'FontSize', 14, 'FontWeight', 'bold');

fprintf('\n✓ Visualisasi selesai\n');