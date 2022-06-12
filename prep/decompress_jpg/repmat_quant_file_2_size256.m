% Method 1: using jpeg pictures, repmat the 8X8 quantization table to 256x256
i = jpeg_read('./datasets/alaskav2/ALASKA_v2_JPG_256_QF95_GrayScale/80005.jpg');        % Read a picture
% i = jpeg_read('xxx.jpg');        % Read a picture
k = i.quant_tables{1};    % Get a quantization table
quant = repmat(k,32,32);  % Repmat to 256*256
save('quant_95.mat','quant');% Save as quant.mat file

% % Method 2: according to the quantization table with 8x8 first
% quant_file = load('quant_75.mat'); % Read the 8X8 quantization table
% quant = repmat(quant_file.quant,32,32); % repmat to 256*256
% save('quant.mat','quant');