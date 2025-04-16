% 参数设置
sigma = 25;       % 噪声标准差（根据实际噪声调整，对应0-255范围）
profile = 'np';   % 使用默认配置'np'（正常模式）
print_to_screen = 1; % 显示处理信息

% 处理所有三张图像
for img_num = 1:3
    % 读取图像（自动转换为灰度）
    filename = sprintf('image%d.png', img_num);
    z = im2double(imread(filename));
    
    % 如果是彩色图像，取消下行注释转换为灰度
    % z = rgb2gray(z); 
    
    % 执行BM3D去噪
    [~, y_est] = BM3D(1, z, sigma, profile, print_to_screen);
    
    % 保存结果
    imwrite(y_est, sprintf('denoised_image%d.png', img_num));
    fprintf('已保存去噪图像: denoised_image%d.png\n', img_num);
end
