% SETUP VERIFICATION & INFO
% File ini untuk verifikasi bahwa semua komponen siap

% ===== STRUKTUR PROJECT =====
% ML/
%  ├── sample/                                (Dataset - SUDAH ADA)
%  │   ├── immature/                         (128 gambar)
%  │   ├── semi-mature/                      (110 gambar)
%  │   └── mature/                           (185 gambar)
%  │
%  ├── training.m                            (✓ Updated)
%  ├── interface.m                           (✓ Updated)
%  ├── predict_single_image.m               (✓ Updated)
%  ├── trainedKNN_Blueberry.mat             (Generated after training)
%  ├── README.md                            (✓ Updated)
%  ├── QUICK_START.txt                      (✓ Created)
%  └── .github/copilot-instructions.md      (✓ Existing)


% ===== QUICK CHECK =====
% Jalankan command berikut di MATLAB console untuk verify:

% Check current directory
pwd

% Check if sample folder exists
if isdir('sample')
    disp('✓ Folder sample ditemukan');
    if isdir('sample/immature')
        disp('  ✓ sample/immature/ OK');
    end
    if isdir('sample/semi-mature')
        disp('  ✓ sample/semi-mature/ OK');
    end
    if isdir('sample/mature')
        disp('  ✓ sample/mature/ OK');
    end
else
    disp('❌ Folder sample TIDAK ditemukan!');
end

% List files
fprintf('\nGambar per kategori:\n');
fprintf('  immature:     %d file\n', length(dir('sample/immature/*.jpg')));
fprintf('  semi-mature:  %d file\n', length(dir('sample/semi-mature/*.jpg')));
fprintf('  mature:       %d file\n', length(dir('sample/mature/*.jpg')));


% ===== READY TO RUN =====
% Semua file sudah siap. Jalankan:
%
%  >> training              (untuk training model)
%  >> Interface             (untuk GUI prediksi)
%  >> predict_single_image  (untuk prediksi single image)
