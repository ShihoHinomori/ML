function Interface
    close all;

    % ===== SETUP PATH =====
    currentFile = mfilename('fullpath');
    projectDir = fileparts(currentFile);
    modelPath = fullfile(projectDir, 'trainedKNN_Blueberry.mat');
    
    % ===== LOAD MODEL =====
    if ~isfile(modelPath)
        fprintf('\n❌ ERROR: Model tidak ditemukan!\n');
        fprintf('Path: %s\n\n', modelPath);
        fprintf('Solusi: Jalankan training.m terlebih dahulu\n');
        errordlg(sprintf('Model tidak ditemukan!\n\nJalankan training.m terlebih dahulu.\n\nPath: %s', modelPath), 'Error');
        return;
    end
    
    load(modelPath, 'mdl');
    fprintf('✓ Interface GUI berhasil dimuat\n');
    fprintf('✓ Model KNN siap digunakan\n');

    % ===== CREATE MAIN FIGURE =====
    guiData = struct();
    guiData.model = mdl;
    guiData.projectDir = projectDir;
    guiData.currentImage = [];
    guiData.currentImagePath = '';
    guiData.showAll = false;

    mainFig = figure('Name', 'Klasifikasi Kematangan Blueberry - KNN', ...
        'NumberTitle', 'off', 'MenuBar', 'none', ...
        'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8], ...
        'Color', [0.95 0.95 0.95]);

    % ===== CREATE UI COMPONENTS =====
    
    % Title
    uicontrol(mainFig, 'Style', 'text', 'Position', [20, 750, 600, 40], ...
        'String', 'KLASIFIKASI KEMATANGAN BLUEBERRY - KNN', ...
        'FontSize', 14, 'FontWeight', 'bold', ...
        'BackgroundColor', [0.95 0.95 0.95], 'HorizontalAlignment', 'left');

    % Buttons
    guiData.btnSelect = uicontrol(mainFig, 'Style', 'pushbutton', ...
        'String', 'Pilih Gambar', 'Position', [20, 700, 120, 35], ...
        'FontSize', 11, 'FontWeight', 'bold', 'BackgroundColor', [0.2 0.7 0.9], ...
        'ForegroundColor', 'w', 'Callback', {@selectImage_Callback, mainFig});

    guiData.btnClear = uicontrol(mainFig, 'Style', 'pushbutton', ...
        'String', 'Hapus Semua', 'Position', [150, 700, 120, 35], ...
        'FontSize', 11, 'FontWeight', 'bold', 'BackgroundColor', [0.9 0.3 0.3], ...
        'ForegroundColor', 'w', 'Callback', {@clearAll_Callback, mainFig});

    % Image axes
    guiData.axOrig = axes(mainFig, 'Position', [0.05 0.45 0.28 0.28]);
    set(guiData.axOrig, 'XTick', [], 'YTick', []);
    title(guiData.axOrig, 'Gambar Asli', 'FontSize', 11, 'FontWeight', 'bold');

    guiData.axProc = axes(mainFig, 'Position', [0.36 0.45 0.28 0.28]);
    set(guiData.axProc, 'XTick', [], 'YTick', []);
    title(guiData.axProc, 'Preprocessing (28×28)', 'FontSize', 11, 'FontWeight', 'bold');

    % Result panel
    guiData.txtPrediction = uicontrol(mainFig, 'Style', 'text', ...
        'Position', [0.67*1000, 620, 250, 50], 'FontSize', 16, 'FontWeight', 'bold', ...
        'String', 'Prediksi: -', 'BackgroundColor', [0.95 0.95 0.95], ...
        'ForegroundColor', [0.2 0.6 0.2], 'HorizontalAlignment', 'center');

    guiData.txtConfidence = uicontrol(mainFig, 'Style', 'text', ...
        'Position', [0.67*1000, 480, 250, 130], 'FontSize', 10, ...
        'String', 'Confidence Score:\n\nimmature: -\nsemi-mature: -\nmature: -', ...
        'BackgroundColor', [0.98 0.98 0.98], 'HorizontalAlignment', 'left', ...
        'VerticalAlignment', 'top');

    % Status bar
    guiData.txtStatus = uicontrol(mainFig, 'Style', 'text', ...
        'Position', [20, 20, 1000, 60], 'FontSize', 10, ...
        'String', 'Siap untuk memproses gambar blueberry. Klik "Pilih Gambar" untuk memulai.', ...
        'BackgroundColor', [0.92 0.92 0.92], 'HorizontalAlignment', 'left', ...
        'VerticalAlignment', 'top');

    guidata(mainFig, guiData);
end

% ===== CALLBACK FUNCTIONS =====

function selectImage_Callback(hObject, eventdata, mainFig)
    guiData = guidata(mainFig);
    
    [filename, pathname] = uigetfile({'*.jpg;*.png;*.bmp', 'Gambar (*.jpg, *.png, *.bmp)'});
    if isequal(filename, 0), return; end

    imagePath = fullfile(pathname, filename);
    
    try
        I = imread(imagePath);
    catch ME
        msgbox(sprintf('Gagal membaca gambar!\n%s', ME.message), 'Error');
        return;
    end

    % Preprocess
    I_proc = imresize(I, [28 28]);
    if size(I_proc, 3) == 3
        I_proc = rgb2gray(I_proc);
    end
    I_proc = im2double(I_proc);

    % Extract HOG & predict
    hogFeat = extractHOGFeatures(I_proc);
    hogFeat = single(hogFeat);
    [label, scores] = predict(guiData.model, hogFeat);

    % Display images
    axes(guiData.axOrig);
    imshow(I);
    title('Gambar Asli', 'FontSize', 11, 'FontWeight', 'bold');

    axes(guiData.axProc);
    imshow(I_proc);
    title('Preprocessing (28×28)', 'FontSize', 11, 'FontWeight', 'bold');

    % Display results
    set(guiData.txtPrediction, 'String', sprintf('Prediksi:\n%s', char(label)));
    
    scoreStr = 'Confidence Score:\n\n';
    classNames = guiData.model.ClassNames;
    for i = 1:length(classNames)
        scoreStr = sprintf('%s%s: %.2f%%\n', scoreStr, string(classNames(i)), scores(i)*100);
    end
    set(guiData.txtConfidence, 'String', scoreStr);

    % Status
    set(guiData.txtStatus, 'String', ...
        sprintf('✓ Gambar: %s\nPrediksi: %s (%.2f%%)', filename, char(label), max(scores)*100));

    guiData.currentImage = I;
    guiData.currentImagePath = imagePath;
    guidata(mainFig, guiData);
end

function clearAll_Callback(hObject, eventdata, mainFig)
    guiData = guidata(mainFig);
    
    cla(guiData.axOrig);
    cla(guiData.axProc);
    set(guiData.axOrig, 'XTick', [], 'YTick', []);
    set(guiData.axProc, 'XTick', [], 'YTick', []);
    
    set(guiData.txtPrediction, 'String', 'Prediksi: -');
    set(guiData.txtConfidence, 'String', 'Confidence Score:\n\nimmature: -\nsemi-mature: -\nmature: -');
    set(guiData.txtStatus, 'String', 'Semua hasil telah dihapus. Klik "Pilih Gambar" untuk memulai lagi.');
    
    guiData.currentImage = [];
    guiData.currentImagePath = '';
    guidata(mainFig, guiData);
end
        'NumberTitle', 'off', 'MenuBar', 'none', ...
        'Units', 'normalized', 'Position', [0.15 0.15 0.7 0.7], ...
        'Color', [0.95 0.95 0.95], 'DeleteFcn', @(~,~) disp('Interface Ditutup'));

    guidata(mainFig, guiData);
    guiData = createUI(mainFig, guiData);
    guidata(mainFig, guiData);
end

function guiData = createUI(mainFig, guiData)
    
    ctrlPanel = uipanel('Parent', mainFig, 'Title', 'Kontrol', ...
        'Units', 'normalized', 'Position', [0.05 0.7 0.25 0.25], ...
        'FontSize', 12, 'FontWeight', 'bold', 'BackgroundColor', [0.9 0.9 0.9]);

    uicontrol(ctrlPanel, 'Style', 'pushbutton', 'String', 'Pilih Gambar', ...
        'Units', 'normalized', 'Position', [0.1 0.6 0.8 0.3], ...
        'BackgroundColor', [0.2 0.7 0.9], 'ForegroundColor', 'w', ...
        'FontSize', 14, 'FontWeight', 'bold', 'Callback', @importImageCallback);

    uicontrol(ctrlPanel, 'Style', 'pushbutton', 'String', 'Hapus Semua', ...
        'Units', 'normalized', 'Position', [0.1 0.2 0.8 0.3], ...
        'BackgroundColor', [0.9 0.3 0.3], 'ForegroundColor', 'w', ...
        'FontSize', 14, 'FontWeight', 'bold', 'Callback', @clearAllCallback);

    resultPanel = uipanel('Parent', mainFig, 'Title', 'Hasil Prediksi', ...
        'Units', 'normalized', 'Position', [0.05 0.25 0.25 0.4], ...
        'FontSize', 12, 'FontWeight', 'bold', 'BackgroundColor', [0.9 0.9 0.9]);

    guiData.predText = uicontrol(resultPanel, 'Style', 'text', ...
        'String', 'Prediksi: -', 'FontSize', 16, 'FontWeight', 'bold', ...
        'Units', 'normalized', 'Position', [0.05 0.85 0.9 0.12], ...
        'BackgroundColor', [0.9 0.9 0.9]);

    guiData.detailText = uicontrol(resultPanel, 'Style', 'text', ...
        'String', 'Top 3 Hasil:', 'FontSize', 11, 'FontWeight', 'bold', ...
        'Units', 'normalized', 'Position', [0.05 0.75 0.9 0.08], ...
        'BackgroundColor', [0.9 0.9 0.9], 'HorizontalAlignment', 'left');

    guiData.resultsText = uicontrol(resultPanel, 'Style', 'text', ...
        'String', '-', 'FontSize', 10, ...
        'Units', 'normalized', 'Position', [0.05 0.25 0.9 0.5], ...
        'BackgroundColor', [0.9 0.9 0.9], 'HorizontalAlignment', 'left');

    guiData.toggleButton = uicontrol(resultPanel, 'Style', 'pushbutton', ...
        'String', 'Tampilkan Semua Hasil ▼', 'FontSize', 9, ...
        'Units', 'normalized', 'Position', [0.05 0.05 0.9 0.15], ...
        'BackgroundColor', [0.7 0.7 0.7], 'Callback', @toggleResultsCallback);

    guiData.showAll = false;

    imgPanel = uipanel('Parent', mainFig, 'Title', 'Pemrosesan Gambar', ...
        'Units', 'normalized', 'Position', [0.35 0.05 0.6 0.9], ...
        'FontSize', 12, 'FontWeight', 'bold', 'BackgroundColor', [0.9 0.9 0.9]);

    guiData.axesOriginal = axes('Parent', imgPanel, 'Position', [0.1 0.65 0.35 0.3]);
    title(guiData.axesOriginal, 'Gambar Asli', 'FontSize', 11, 'FontWeight', 'bold');
    
    guiData.axesPreprocessed = axes('Parent', imgPanel, 'Position', [0.55 0.65 0.35 0.3]);
    title(guiData.axesPreprocessed, 'Grayscale', 'FontSize', 11, 'FontWeight', 'bold');
    
    guiData.axesFinal = axes('Parent', imgPanel, 'Position', [0.3 0.25 0.4 0.35]);
    title(guiData.axesFinal, 'Final (28x28)', 'FontSize', 11, 'FontWeight', 'bold');

end

function importImageCallback(~, ~)
    [file, path] = uigetfile({'*.png;*.jpg;*.jpeg;*.bmp;*.tiff'}, 'Pilih Gambar Blueberry');
    if isequal(file, 0), return; end

    I = imread(fullfile(path, file));
    guiData = guidata(gcf);
    guiData.currentImage = I;
    guidata(gcf, guiData);

    processAndDisplayImage(I);
end

function clearAllCallback(~, ~)
    guiData = guidata(gcf);
    cla(guiData.axesOriginal);
    cla(guiData.axesPreprocessed);
    cla(guiData.axesFinal);
    
    title(guiData.axesOriginal, 'Gambar Asli', 'FontSize', 11, 'FontWeight', 'bold');
    title(guiData.axesPreprocessed, 'Grayscale', 'FontSize', 11, 'FontWeight', 'bold');
    title(guiData.axesFinal, 'Final (28x28)', 'FontSize', 11, 'FontWeight', 'bold');

    set(guiData.predText, 'String', 'Prediksi: -');
    set(guiData.detailText, 'String', 'Top 3 Hasil:');
    set(guiData.resultsText, 'String', '-');
    set(guiData.toggleButton, 'String', 'Tampilkan Semua Hasil ▼');
    
    guiData.currentImage = [];
    guiData.showAll = false;
    guidata(gcf, guiData);
end

function toggleResultsCallback(~, ~)
    guiData = guidata(gcf);
    guiData.showAll = ~guiData.showAll;
    
    if guiData.showAll
        set(guiData.detailText, 'String', 'Semua Hasil:');
        set(guiData.toggleButton, 'String', 'Tampilkan Lebih Sedikit ▲');
    else
        set(guiData.detailText, 'String', 'Top 3 Hasil:');
        set(guiData.toggleButton, 'String', 'Tampilkan Semua Hasil ▼');
    end
    
    guidata(gcf, guiData);
    
    if ~isempty(guiData.currentImage)
        processAndDisplayImage(guiData.currentImage);
    end
end

function processAndDisplayImage(I)
    guiData = guidata(gcf);
    
    if size(I,3) == 3
        I_gray = rgb2gray(I);
    else
        I_gray = I;
    end
    
    I_gray = im2double(I_gray);
    I_final = imresize(I_gray, [28 28]);
    
    imshow(guiData.currentImage, 'Parent', guiData.axesOriginal);
    imshow(I_gray, 'Parent', guiData.axesPreprocessed);
    imshow(I_final, 'Parent', guiData.axesFinal);

    hog = extractHOGFeatures(I_final);
    hog = single(hog);
    
    [label, scores] = predict(guiData.model, hog);
    
    [sortedScores, sortedIndices] = sort(scores, 'descend');
    classLabels = guiData.model.ClassNames;
    
    set(guiData.predText, 'String', sprintf('Prediksi: %s', string(label)));
    
    resultsStr = '';
    if guiData.showAll
        validResults = sortedScores >= 0.1;
        numResults = sum(validResults);
    else
        numResults = min(3, length(sortedScores));
    end
    
    for i = 1:numResults
        if guiData.showAll && sortedScores(i) < 0.1
            break;
        end
        confidence = sortedScores(i) * 100;
        ripeness = classLabels(sortedIndices(i));
        resultsStr = sprintf('%s%s: %.1f%%\n', resultsStr, string(ripeness), confidence);
    end
    
    set(guiData.resultsText, 'String', resultsStr);
end