function file_list = savetheMat(path, file_mask, dpath)   % There should be a'/' at the end of the path
file_path =  path;
img_path_list = dir(strcat(file_path, file_mask));       % Get images of all jpg formats in the folder
img_num = length(img_path_list);                         % Get the total number of images

if ~exist(dpath,'dir'); mkdir(dpath); end

fprintf('totolly %d pictrues\n', img_num);
file_list = cell(img_num, 1);
if img_num > 0 % When there is an image that meets the conditions

    for j = 1:img_num % Read images one by one
        image_name = img_path_list(j).name;% Image name
        a = jpeg_read(strcat(file_path,image_name));
        im = a.coef_arrays{1,1};
        save(strcat(dpath,image_name(1:end-4),'.mat'),'im');
        fprintf('completed： %s\n', strcat(dpath,image_name(1:end-4),'.mat'));% Print the scanned image path name
    end
end
end



% Example of Use：
% f = savetheMat('./alaskav2/ALASKA_v2_JPG_256_QF75_GrayScale/','*.jpg','./alaskav2/jpeg-mat/qf75/');