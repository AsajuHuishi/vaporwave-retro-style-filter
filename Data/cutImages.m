%% Images 1080x1920->ImagesCut 1080x1428
for i=2:543
    i
    pre = imread(['./Images/',num2str(i),'.jpg']);
    ret = pre(:,242:1669,:);
%     figure(3);
%     imshow(ret);
    imwrite(ret,['./ImagesCut/',num2str(i),'.jpg'])
end

