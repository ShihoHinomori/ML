# Copilot Instructions - Blueberry Ripeness Classification

## Project Overview
This is a MATLAB-based blueberry ripeness classification system using K-Nearest Neighbors (KNN) with HOG (Histogram of Oriented Gradients) feature extraction. The system classifies blueberries into three ripeness stages: **immature**, **semi-mature**, and **mature**. The project contains three executable components: training pipeline, single image prediction, and interactive GUI.

## Architecture & Data Flow

### Core Pipeline
1. **Training** (`training.m`): Loads blueberry images from `sample/` subdirectories (immature, semi-mature, mature), preprocesses to 28×28 grayscale, extracts HOG features, trains KNN model (k=5), saves to `trainedKNN_Blueberry.mat`
2. **Prediction** (`predict_single_image.m`): Loads trained model, preprocesses input blueberry image, extracts HOG features, predicts ripeness stage with confidence scores
3. **GUI Interface** (`interface.m`): Interactive tool for batch blueberry testing with confidence scores and detailed result visualization in Indonesian

### Ripeness Classification
- **Immature** (Belum Matang): Green/early stage blueberries
- **Semi-mature** (Setengah Matang): Transitional stage with color development  
- **Mature** (Matang/Sangat Matang): Fully ripe, dark purple/black blueberries

### Critical Data Flow
- **Input**: RGB/Grayscale blueberry images of any size in `sample/{ripeness_stage}/` folders
- **Preprocessing**: All images standardized to 28×28, converted to grayscale, normalized to double [0,1]
- **Feature Extraction**: HOG with default MATLAB parameters returns feature vector of fixed length
- **Model Storage**: Binary `.mat` file containing trained KNN model object (`mdl`)

## Key Development Patterns

### Image Processing Convention
All image manipulation follows this pattern:
```matlab
I = imresize(I, [28 28]);           % Always resize to 28x28
if size(I,3) == 3
    I = rgb2gray(I);                 % Convert RGB→grayscale (removes color info)
end
I = im2double(I);                    % Normalize to [0,1]
```
This preprocessing is duplicated in training, prediction, and GUI - maintain consistency when modifying.

### HOG Feature Extraction
```matlab
hog = extractHOGFeatures(I_final);   % Returns 1D vector
hog = single(hog);                   % Cast to single for model input
```
The feature length is determined dynamically from first image (`hogLength = length(hogFeat)`) - ensures compatibility if HOG parameters change.

### KNN Model Usage
- **Training**: `fitcknn(Xtrain, Ytrain, 'NumNeighbors', 5, 'Distance', 'euclidean', 'Standardize', true)`
- **Prediction with scores**: `[label, scores] = predict(model, features)` - scores are class posterior probabilities
- **Batch prediction**: `Ypred = predict(model, X)` where X is N×M matrix (N samples, M features)

### GUI State Management
GUI uses `guidata()` for persistent state across callbacks:
```matlab
guiData = guidata(gcf);           % Retrieve
guiData.currentImage = I;
guidata(gcf, guiData);            % Store
```
Key fields: `model`, `currentImage`, `showAll` (toggle state), `axesOriginal/Preprocessed/Final`, `predText/detailText/resultsText` (UI handles).

## Critical Paths & File Dependencies

| File | Purpose | Dependencies | Output |
|------|---------|--------------|--------|
| `training.m` | Build model | `sample/` with immature/semi-mature/mature/ | `trainedKNN_Blueberry.mat` |
| `predict_single_image.m` | Single blueberry prediction | `trainedKNN_Blueberry.mat` + image file | Console output with confidence |
| `interface.m` | Interactive GUI testing | `trainedKNN_Blueberry.mat` | MATLAB figure + predictions |

**Critical assumption**: `trainedKNN_Blueberry.mat` must exist before running `interface.m` or `predict_single_image.m`. Both check with `if ~isfile(modelPath)` and error if missing.

## Evaluation & Metrics
Classification metrics computed in `training.m`:
- **Accuracy**: `mean(Ypred == Ytest)`
- **Per-stage**: Precision, Recall, F1-score derived from confusion matrix for each ripeness stage
- **Display**: Classification report mimics scikit-learn format with macro/weighted averages
- **Visualization**: `confusionchart()` plots confusion matrix

## Common Workflows

### Training from blueberry dataset
1. Organize images into `sample/immature/`, `sample/semi-mature/`, `sample/mature/`
2. Run `training.m` - automatically discovers classes via `imageDatastore(..., 'LabelSource', 'foldernames')`
3. Model saved as `trainedKNN_Blueberry.mat`

### Testing single blueberry image
- **Single image**: Run `predict_single_image.m`, modify `imagePath` variable to your blueberry image
- **Batch GUI testing**: Run `Interface()`, use "Pilih Gambar" button, toggle "Tampilkan Semua Hasil" for confidence details
- **Output includes**: Predicted ripeness stage + confidence scores for all three stages

### Modifying model parameters
- **k-value**: Change `defaultK = 5` in `training.m` (try 3-7 for blueberry classification)
- **Image size**: Modify `[28 28]` resize operations (affects HOG feature length, retrain required)
- **Distance metric**: Edit `'Distance', 'euclidean'` in `fitcknn()` call (consider 'cityblock' for blueberry features)

## Language & Locale Notes
Project uses Indonesian UI and console output (e.g., "Gambar Asli", "Prediksi", "Tingkat Kematangan"). All user-facing strings in GUI are Indonesian for local accessibility. Error messages are contextual; maintain this pattern when extending.
