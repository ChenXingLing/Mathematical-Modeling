im=imread('peppers.png');
%im=imread('src/70054356_p0.png');
%im=imread('src/89911742_p0.png');
im=double(im);[m,n,r] = size(im);

%sz=size(im)
%rk=[rank(im(:,:,1)),rank(im(:,:,2)),rank(im(:,:,3))]
rk=rank(im(:,:,1));

figure;imshow(uint8(im));title(['原图: ',num2str(m),'×',num2str(n)]);

fprintf("图像大小%d×%d\n奇异值选取数    压缩率        误差    \n",m,n);
K=floor(0.5*(sqrt(m*m+n*n+6*m*n)-m-n));
x=1:K;y1=double(x);y2=double(x);
%for k=floor(rk*0.25):floor(rk*0.25)
for k=1:K
    img=zeros(m,n,r);
    err=0;sum=0;
    for i=1:3
        [U,S,V]=svd(im(:,:,i));
        img(:,:,i)=U(:,1:k)*S(1:k,1:k)*transpose(V(:,1:k));
        err=err+norm(im(:,:,i)-img(:,:,i));
        sum=sum+norm(im(:,:,i));
    end
    
    rate=(m*k+k*k+k*n)/(m*n)*100;
    error=err/sum*100;
    y1(k)=rate;
    y2(k)=error;
    fprintf("    %2d         %.2f%%     %.6f%%\n",k,rate,error);
    
    %if k==floor(rk*0.25)
    if k==10|k==50|k==100|k==150
        figure;imshow(uint8(img));
        title(['压缩图像: ',num2str(m),'×',num2str(n),'奇异值选取数: ',num2str(k),'  压缩率: ',num2str(rate),'%', '  误差: ',num2str(error),'%']);
        %break
    end
end
figure;plot(x,y1);xlabel('奇异值选取数 k');ylabel('压缩率 rate(%)');
figure;plot(x,y2);xlabel('奇异值选取数 k');ylabel('误差 error(%)');
figure;plot(y1,y2);xlabel('压缩率 rate(%)');ylabel('误差 error(%)');

