function file_list = resize_compression(path, file_mask, dpath)
file_path =  path;
img_path_list = dir(strcat(file_path, file_mask)); 
img_num = length(img_path_list);                   % Get the total number of images
fprintf('totolly %d pictrues\n', img_num);

file_list = cell(img_num, 1);
if img_num > 0                                     % When there is an image that meets the conditions
    for j = 1:img_num                              % Read images one by one
        image_name = img_path_list(j).name;        % Image name
        fprintf('picname %s', image_name);
        a = imread(strcat(file_path,image_name));
        a = imresize(a,[256,256]);
        imwrite(a, strcat(dpath,image_name(1:end-4),'.jpg'),'jpg','quality',95);
        fprintf('completed： %s\n', strcat(dpath,image_name(1:end-4),'.jpg'));  % Print the scanned image path name
    end
end
end


% a = resize_compression('./datasets/BOSS_BOWS2/Boss_256/','*.pgm','./datasets/BOSS_BOWS2/Boss_256_qf75/');
