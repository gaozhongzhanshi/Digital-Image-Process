clc;
clear;
I = imread('lena.bmp');
[M,N]=size(I);

imshow(I)
imshow(I,[LOW HIGH])
imshow(RGB)
imshow(BW)

imtool
% imtool(I)
% imtool(I,[LOWHIGH])
A = imread('lady.jpg');
B = impixel(A);

imwrite(A,'lady.jpg')
imwrite(A,'lady.jpg','quality',50)

RGB = imread('flower.jpg');
I = rgb2gray(RGB);
subplot(1,2,1),subimage(RGB)
subplot(1,2,2),subimage(I)

INFO = imfinfo('girl.jpg');
INFO = imfinfo('pinwei.bmp');

A = imread('lotus.jpg');
[x,y,z] = sphere;
warp(x,y,z,A)

I = imread('car.jpg');
x = imcrop(I,[1 89 512 512]);
imshow(x)

A=imread('lena.bmp');
B1=imresize(A,1.5); % 1.5倍
B2=imresize(A,[420 384]); % 非均匀
C1=imresize(A,0.7); % 0.7倍
C2=imresize(A,[150 180]); % 非均匀
figure;
imshow(B1);
title('均匀放大图');
figure;
imshow(B2);
title('非均匀放大图图');
figure;
imshow(C1);
title('均匀缩小图');
figure;
imshow(C2);
title('非均匀缩小图');

I = imread('lena.bmp');
J = imrotate(I,45); % 45°
K = imrotate(I,90); % 90°
subplot(1,3,1);
imshow(I);
subplot(1,3,2);
imshow(J);
subplot(1,3,3);
imshow(K);

I = im2double(imread('bird.jpg'));
NewImage1=imrotate(I,15,'bilinear'); % 剪切
NewImage2=imrotate(I,15,'bilinear','crop'); % 不剪切
subplot(131),imshow(I),title('原图');
subplot(132),imshow(NewImage1),title('剪切');
subplot(133),imshow(NewImage2),title('不剪切');

Image = imread('flower.jpg');
H = flipdim(Image,2); % 水平镜像
V = flipdim(Image,1); % 垂直镜像
C = flipdim(H,1); % 对角镜像
NI = [Image H; V C];
subplot(221),imshow(Image),title('原图');
subplot(222),imshow(H),title('水平镜像');
subplot(223),imshow(V),title('垂直镜像');
subplot(224),imshow(C),title('对角镜像');
figure,imshow(NI),title('图像拼接');

%实验二课后
Image = imread('man.jpg');
A = imrotate(Image,45); % 旋转
B = imresize(A,1.5); % 缩放
C = imrotate(Image,60); % 旋转
D = imresize(A,0.9); % 旋转
imshow(B)

Image = imread('man.jpg');
A = imrotate(Image,45); % 旋转
se = translate(strel(1),[50 50]); % 平移
B = imresize(A,1.5); % 缩放
C = imresize(A,1.5); % 缩放
se = translate(strel(1),[50 50]); % 平移
D = imdilate(C,se);
E = imrotate(C,30); % 旋转
imshow(E)

I = imread('fruit.jpg');
A = [1 0.5 0; 0 1 0; 0 0 1]; % 变换矩阵
tform = affine2d(A); % 仿射矩阵
J = imwarp(I,tform);
imshow(I)

I = imread('couple.bmp');
J = I*tan(pi/3); % 设置角度值
imshow(J)
imshow(I)

I=imread('couple.bmp');
J=I*tan(pi/6)+100;
imshow(J)

J = imcomplement(I);

I=imread('lena.bmp');
J=im2bw(I,0.7); % 0.7范围
imshow(J)
imshow(I)

I=imread('lena.bmp');
imshow(I);
[M,N]=size(I); % 拿到图像尺寸
for i= 1:M % 双层循环遍历每个像素
    for j= 1:N
        if I(i,j)<=30 I(i,j)=I(i,j);
        elseif I(i,j)<=150 I(i,j) = (200-30)/(150-30)*(I(i,j)-30) + 30;
        else I(i,j) = (255-200)/(255-150)*(I(i,j)-150)+200;
        end
    end
end
figure(2)
imshow(I)

I=imread('lena.bmp');
figure, imshow(I);
[M,N]=size(I); % 尺寸获取
for i=1:M % 遍历像素
    for j=1:N
        if I(i, j) <= 50 I(i, j) = 40;  % 灰度变换
        elseif I(i, j) <= 180 I(i, j) = 220;
        else I(i, j) = 40;
        end
    end
end
figure; imshow(I)

I=imread('lena.bmp');
% low为0.3 high为0.7
J=imadjust(I,[0.3 0.7],[ ],1.5); 
figure, imshow(I);
figure, imshow(J)

% 对数变换
I=imread('lena.bmp');
J=double(I);
K=J/255;
C=2;
Z=C*log(1+K);
imshow(Z)

% γ变换
I=imread('lena.bmp');
J=double(I);
K=J/255;
gamma=4;
Z=K.^gamma;
imshow(Z)

I=imread('plane1.jpg');
imshow(I);
[M,N]=size(I); % 拿到图像尺寸
for i= 1:M % 双层循环遍历每个像素
    for j= 1:N
        if I(i,j)<=40 I(i,j)=I(i,j);
        elseif I(i,j)<=160 I(i,j) = (220-40)/(160-40)*(I(i,j)-40) + 40;
        else I(i,j) = (255-200)/(255-150)*(I(i,j)-150)+200;
        end
    end
end
figure(2)
imshow(I)

% 灰度切分
I=imread('plane1.jpg');
figure, imshow(I);
[M,N]=size(I); % 尺寸获取
for i=1:M % 遍历像素
    for j=1:N
        if I(i, j) <= 50 I(i, j) = I(i,j);  % 灰度变换
        elseif I(i, j) <= 180 I(i, j) = 220;
        else I(i, j) = I(i,j);
        end
    end
end
figure; imshow(I)


I=imread('plane1.jpg');
% low为0.3 high为0.7
J=imadjust(I,[0.4 0.8],[ ],1.5); 
figure, imshow(I);
figure, imshow(J)

% 将沙漠和车相加
I=imread('desert.jpg');
J=imread('car.jpg');
K=imadd(I,J);
imshow(K)

I=imread('lena.bmp');
J=imadd(I,100);
subplot(121),imshow(I)
subplot(122),imshow(J)


A1 = imread('desert.jpg');
A2 = imread('car.jpg');
Z = imlincomb(0.7,A1,0.3,A2);
imshow(Z)

% 加上Gaussian噪声，相加降低噪声
I=imread('lena.bmp');
J=imnoise(I,'gaussian',0,0.02); % 加入噪声
subplot(121),imshow(I);
subplot(122),imshow(J);
[M,N]=size(I);
K=zeros(M,N);
for i=1:100
    J=imnoise(I,'gaussian',0,0.02);
    J1=im2double(J);
    K=K+J1;
end
% 相加100次后求均值
K=K/100;
figure,imshow(K)

I=imread('desert.jpg');
J=imread('car.jpg');
K=imsubtract(I,J);
imshow(K)

I=imread('lena.bmp');
J=imsubtract(I,100);
subplot(121),imshow(I)
subplot(122),imshow(J)

I=imread('hallback.bmp');
J=imread('hallforeground.bmp');
K=imabsdiff(I,J);
imshow(K)

I=imread('lena.bmp');
J=immultiply(I,2);
subplot(121),imshow(I)
subplot(122),imshow(J)

I=imread('bird.jpg');
J=imread('birdtemplet.bmp');
K=immultiply(I,J);
imshow(K)

I=imread('bird.jpg');
J=imread('birdtemplet.bmp');
K=imdivide(I,J);
imshow(K)

I=imread('lena.bmp');
J=imdivide(I,2);
subplot(121),imshow(I)
subplot(122),imshow(J)

I = imread('bird.jpg');
J = imread('birdtemplet.bmp');
K = xor(I,J);
imshow(double(K))
K = not(I);
imshow(double(K))
K = not(I);
K = ~I;
K = and(I,J);
K = or(A,B);
K = A|B;
K = xor(I,J);
imshow(double(K))

I=imread('lena.bmp');
[M,N]=size(I);
J=ones(M,N)*255; J(20:150,100:200)=0;
for i=1:M
    for j=1:N
        K(i, j) = bitor(I(i, j), J(i, j));
    end
end
subplot(221); imshow(I);
subplot(222); imshow(J);
subplot(223); imshow(K)

I=imread('lena.bmp');
[M,N]=size(I);
J=ones(M,N)*255; J(20:150,100:200)=0;
for i=1:M
    for j=1:N
        K(i, j) = bitxor(I(i, j), J(i, j));
    end
end
subplot(221); imshow(I);
subplot(222); imshow(J);
subplot(223); imshow(K)

I=imread('lena.bmp');
figure
subplot(121),imhist(I);
subplot(122),imhist(I,64);

X=imread('peppers.jpg');
Y=rgb2gray(X);
imhist(Y)

X=imread('peppers.jpg');
R=X(:,:,1);
G=X(:,:,2);
B=X(:,:,3);
subplot(221),imshow(X),title('彩色原图');
subplot(222),imhist(R),title('红色通道直方图');
subplot(223),imhist(G),title('绿色通道直方图');
subplot(224),imhist(B),title('蓝色通道直方图');

I=imread('lena.bmp');
[M,N]=size(I);
[counts1,x]=imhist(I,128);
subplot(121),stem(x,counts1);
counts2=counts1/M/N;
subplot(122),stem(x,counts2);

I=imread('lena.bmp');
S=numel(I);  % 求像素总数
Pr=imhist(I)/S;
k=0:255;
figure,stem(k,Pr);

I=imread('lena.bmp');
[counts,binlocations]=imhist(I);

X=imread('lena.bmp');
edges=[0 10:2:150 255];
histogram(X,edges),xlabel('灰度级'),ylabel('像素数');
figure,imhist(X);

[n,edges]=histcounts(X);

I=imread('lena.bmp');
subplot(221),imshow(I),title('原始图像');
subplot(223),imhist(I),title('原始图像直方图');
J=histeq(I,256);
subplot(222),imshow(J),title('均衡化图像');
subplot(224),imhist(J),title('均衡化图像的直方图');

I=imread('lena.bmp');
imshow(I)
figure,imhist(I)
J=histeq(I);
figure,imhist(J)
figure,imshow(J)

[Y,T]=histeq(X);

X=imread('lena.bmp');
REF=imread('cameraman.jpg');
[HGRAM,x]=imhist(REF);
Y=histeq(X,HGRAM);
subplot(231),imshow(X),title('原始图像');
subplot(234),imhist(X),title('原始图像直方图');
subplot(232),imshow(REF),title('参考图像');
subplot(235),imhist(REF),title('参考图像直方图');
subplot(233),imshow(Y),title('输出图像');
subplot(236),imhist(Y),title('输出图像直方图');

I=imread('lena.bmp');
Isp=imnoise(I,'salt & pepper',0.01);
Ig=imnoise(I,'gaussian',0.01);
subplot(131),imshow(I),title('原始图像');
subplot(132),imshow(Isp),title('添加椒盐噪声');
subplot(133),imshow(Ig),title('添加高斯噪声');

I=imread('lena.bmp');
X=imnoise(I,'salt & pepper',0.01);
w1=[1 1 1; 1 1 1; 1 1 1]/9;
w2=fspecial('average',3);
w3=fspecial('average',5); % 均值滤波器
w4=fspecial('average',7);
Y1=imfilter(X,w1,'conv','replicate');
Y2=imfilter(X,w2,'conv','replicate');
Y3=imfilter(X,w3,'conv','replicate');
Y4=imfilter(X,w4,'conv','replicate');
subplot(221),imshow(Y1),title('3*3均值滤波');
subplot(222),imshow(Y2),title('3*3均值滤波');
subplot(223),imshow(Y3),title('5*5均值滤波');
subplot(224),imshow(Y4),title('7*7均值滤波');

I=im2double(imread('lena.bmp'));
Isp=imnoise(I,'salt & pepper',0.01);
Ig=imnoise(I,'gaussian',0.01);
result1=filter2(fspecial('average',3),Isp);
result2=filter2(fspecial('average',5),Isp);
result3=filter2(fspecial('average',3),Ig);
result4=filter2(fspecial('average',5),Ig);
subplot(221),imshow(result1),title('3*3模板抑制椒盐噪声');
subplot(222),imshow(result2),title('5*5模板抑制椒盐噪声');
subplot(223),imshow(result3),title('3*3模板抑制高斯噪声');
subplot(224),imshow(result4),title('5*5模板抑制高斯噪声');

I=imread('lena.bmp');
imshow(I,[ ]);
J=imnoise(I,'salt & pepper',0.01);
figure,imshow(J);
h0=1/9.*[1 1 1 1 1 1 1 1 1];
h1=[0.1 0.1 0.1; 0.1 0.2 0.1; 0.1 0.1 0.1];
h2=1/16.*[1 2 1 2 4 2 1 2 1];
h3=1/8.*[1 1 1; 1 0 1; 1 1 1];
g0=filter2(h0,J);
g1=filter2(h1,J);
g2=filter2(h2,J);
g3=filter2(h3,J);
figure,imshow(g0,[ ]);
figure,imshow(g1,[ ]);
figure,imshow(g2,[ ]);
figure,imshow(g3,[ ]);

I=imread('lena.bmp');
X=imnoise(I,'salt & pepper',0.01);
Y1=medfilt2(X,[5 5]);
imshow(Y1),title('5*5中值滤波');


I=imread('cameraman.jpg');
subplot(131),imshow(I);
H=fspecial('sobel');
H=H'; 
J=filter2(H,I);
subplot(132),imshow(J);
H=H';
J=filter2(H,I);
subplot(133),imshow(J);

I=imread('cameraman.jpg');
subplot(131),imshow(I),title('原图像');
Laplace=[0 -1 0; -1 4 -1; 0 -1 0];
LapI=imfilter(I,Laplace,'conv','replicate');
subplot(132),imshow(LapI),title('Laplace图像');
Y=imadd(I,LapI);
subplot(133),imshow(Y),title('锐化图像');

B=medfilt2(A);   
B=medfilt2(A,[M N]);

I=im2double(imread('lena.bmp'));
BW=edge(I,'roberts');
H1=[1 0; 0 -1];
H2=[0 1;-1 0];
R1=imfilter(I,H1);
R2=imfilter(I,H2);
edgeI=abs(R1)+abs(R2);
sharpI=I+edgeI;
subplot(221),imshow(I),title('原图');
subplot(222),imshow(edgeI),title('roberts梯度图');
subplot(223),imshow(BW),title('roberts边缘检测');
subplot(224),imshow(sharpI),title('roberts锐化图像');

I=imread('cameraman.jpg');
DFT=fft2(I);
ADFT=abs(DFT); % 计算傅里叶
top=max(ADFT(:));
bottom=min(ADFT(:));
ADFT1=(ADFT-bottom)/(top-bottom)*100; % 规格化
ADFT2=fftshift(ADFT);
subplot(131),imshow(I),title('原图');
subplot(132),imshow(ADFT1),title('原频谱图');
subplot(133),imshow(ADFT2),title('移位频谱图');

I=imread('lena.bmp');
figure,imshow(I);
F1=fft2(I); 
% 对数变换增强效果
figure,imshow(log(abs(F1)+1),[0 10]); 
F2=fftshift(F1);
figure,imshow(log(abs(F2)+1),[0 10]);

I=imread('desert.jpg');
DFT=fftshift(fft2(I));
DFT=abs(DFT);
top=max(DFT(:));
bottom=min(DFT(:));
ADFT=(DFT-bottom)/(top-bottom)*100;
subplot(121),imshow(I),title('原图');
subplot(122),imshow(ADFT),title('彩色频谱图');

I=imread('peppers.jpg');
DFT=fftshift(fft2(I));
ADFT=abs(DFT);
top=max(ADFT(:));
bottom=min(ADFT(:));
ADFT=(ADFT-bottom)/(top-bottom)*100;
reI=ifft2(ifftshift(DFT));
reI=abs(reI);
reI=uint8(reI);
subplot(131),imshow(I),title('原图');
subplot(132),imshow(ADFT),title('彩色频谱图');
subplot(133),imshow(reI),title('重构图');

f=zeros(30,30);
f(5:24,13:17)=1;
figure,imshow(f,'InitialMagnification','fit');
F=fft2(f);
F2=log(abs(F));
figure,imshow(F2,[-1 5],'InitialMagnification','fit');colormap(jet);
F=fft2(f,256,256);
figure,imshow(log(abs(F)),[-1 5],'InitialMagnification','fit');colormap(jet);
F2=fftshift(F);
figure,imshow(log(abs(F2)),[-1 5],'InitialMagnification','fit');colormap(jet);

% 读取一张彩色图像
img = imread('lady.jpg');
% 将彩色图像转换为灰度图像
gray_img = rgb2gray(img);
% 显示原始图像
figure;
subplot(1, 2, 1);
imshow(gray_img);
title('Original Image');
% 对灰度图像进行DCT变换
dct_img = dct2(double(gray_img));
% 保留部分DCT系数（压缩）
compression_ratio = 0.1; % 保留10%的DCT系数
num_coeffs = round(compression_ratio * numel(dct_img));
dct_compressed = dct_img;
dct_compressed(num_coeffs+1:end) = 0;
% 对压缩后的DCT系数进行IDCT反变换
reconstructed_img = uint8(idct2(dct_compressed));
% 显示重构后的图像
subplot(1, 2, 2);
imshow(reconstructed_img);
title('Reconstructed Image');

I=imread('lena.bmp');
Ig=imnoise(I,'gaussian');
subplot(121),imshow(Ig),title('噪声图像');
FI=fftshift(fft2(double(Ig)));
subplot(122),imshow(log(abs(FI)),[ ]),title('傅里叶频谱');
[N,M]=size(FI);
g=zeros(N,M);
r1=floor(M/2);
r2=floor(N/2);
len=sqrt(r1^2+r2^2);
% 截止频率
D0=[0.05*len 0.1*len 0.2*len 0.5*len];
for i=1:4
    for x=1:M
        for y=1:N
            D=sqrt((x-r1)^2+(y-r2)^2);
            if D <= D0(i)
                h=1;
            else
                h=0;
            end
            g(y,x)=h*FI(y,x);
        end
    end
g=real(ifft2(ifftshift(g)));
figure,imshow(uint8(g)),title(['理想低通滤波D0=',num2str(D0(i))]);
end

I=imread('lena.bmp');
Ig=imnoise(I,'gaussian');
subplot(121),imshow(Ig),title('噪声图像');
FI=fftshift(fft2(double(Ig)));
subplot(122),imshow(log(abs(FI)),[ ]),title('傅里叶频谱');
[N,M]=size(FI);
g=zeros(N,M);
r1=floor(M/2);
r2=floor(N/2);
len=sqrt(r1^2+r2^2);
D0=[0.05*len 0.1*len 0.2*len 0.5*len];
n=3;
for i=1:4
    for x=1:M
        for y=1:N
            D=sqrt((x-r1)^2+(y-r2)^2);
            h=1/(1+(D/D0(i))^(2*n));
            g(y,x)=h*FI(y,x);
        end
    end
g=real(ifft2(ifftshift(g)));
figure,imshow(uint8(g)),title(['巴特沃斯低通滤波器D0=',num2str(D0(i))]);
end

% 读取一张灰度图像
img = imread('lady.jpg');
img = rgb2gray(img);
% 设计巴特沃斯高通滤波器
D = 0.9; % 截止频率
n = 4; % 阶数
% 计算巴特沃斯高通滤波器系数
[b, a] = butter(n, D, 'high');
filtered_img = filter2(b, a, double(img));
figure;
subplot(1, 2, 1);
imshow(img);
title('Original Image');
subplot(1, 2, 2);
imshow(uint8(filtered_img));
title('Filtered Image');

img = imread('lady.jpg');
img = rgb2gray(img);
% 设计梯形高通滤波器
N = 30; % 滤波器阶数
F = [0.1, 0.2, 0.6, 0.7];
% 计算梯形高通滤波器系数
b = fir1(N, F, 'high');
filtered_img = filter(b, 1, double(img));
figure;
subplot(1, 2, 1);
imshow(img);
title('Original Image');
subplot(1, 2, 2);
imshow(uint8(filtered_img));
title('Filtered Image');

rgb=imread('lotus.bmp');
r=rgb(:,:,1);
g=rgb(:,:,2);
b=rgb(:,:,3);
figure,imshow(r);
figure,imshow(g);
figure,imshow(b);

IR=zeros(128,128);
IR(1:64,1:64)=1;
IG=zeros(128,128);
IG(65:128,1:64)=1;
IB=zeros(128,128);
IB(1:64,65:128)=1;
I=cat(3,IR,IG,IB);
imshow(I);

J=imcomplement(I);

rgb=imread('lotus.bmp');
imshow(rgb);
rgb=im2double(rgb);
r=rgb(:,:,1);
g=rgb(:,:,2);
b=rgb(:,:,3);
I=(r+g+b)/3;
tmp1=min(min(r,g),b);
tmp2=r+g+b;
tmp2(tmp2 == 0) = eps; 
S = 1 - 3. * tmp1./tmp2;
tmp1 = 0.5 * ((r - g) + (r - b));
tmp2=sqrt((r-g).^2+(r-b).*(g-b));
theta=acos(tmp1./(tmp2+eps));
H=theta;
H(b > g) = 2 * pi - H(b > g);
H=H/(2*pi);
H(S==0)=0;
hsi=cat(3,H,S,I);
figure,imshow(H);
figure,imshow(S);
figure,imshow(I);

I=imread('lotus.bmp');
GS8=grayslice(I,8);
GS64=grayslice(I,64);
subplot(131),imshow(I),title('原图');
subplot(132),imshow(GS8,hot(8)),title('8级伪彩色');
subplot(133),imshow(GS64,hot(64)),title('64级伪彩色');

image=rgb2gray(imread('lotus.jpg'));
r=image; g=image; b=image;
r(image < 20) = 0; 
g(image < 20) = 20; 
b(image < 20) = 0;
r(20 < image & image < 40) = 0; 
g(20 < image & image < 40) =50; 
b(20 < image & image < 40) = 0;
r(40 < image & image < 50) = 0; 
g(40 < image & image < 50) =70; 
b(40 < image & image < 50) = 0;
r(50 < image & image < 90) = 30; 
g(50 < image & image < 90) =230; 
b(50 < image & image < 90) = 130;
r(image > 90) = 230; 
g(image > 90) = 230; 
b(image > 90) = 220;
result=cat(3,r,g,b);
imshow(result);

Image=rgb2gray(imread('lotus.jpg'));
r=Image; g=Image; b=Image;
r(Image < 128) = 0;
r(128 <= Image & Image < 192) = 4*Image(128 <= Image & Image <192) - 2 * 255;
r(Image >= 192) = 255;
g(Image < 64) = 4 * Image(Image < 64);
g(64 <= Image & Image < 192) = 255;
g(Image >= 192) = -4 * Image(Image >= 192) + 4 * 255;
b(Image < 64) = 255;
b(64 <= Image & Image < 128) = -4*Image(64 <= Image & Image <128) + 2 * 255;
b(Image >= 128) = 0;
r=uint8(r);
g=uint8(g);
b=uint8(b);
result=cat(3,r,g,b);
imshow(result);

I=imread('lena.bmp');
imwrite(I,'lenac.jpg');
info1=dir('lena.bmp');
b1=info1.bytes;
info2=dir('lenac.jpg');
b2=info2.bytes;
ratio=b1/b2;
figure; imshow('lena.bmp');
figure; imshow('lenac.jpg')

I = imread('lena.bmp');
[h,w]=size(I);
subplot(3,3,1);imshow(I);
title('原始图像');
for k=1:8
    for i=1:h
        for j=1:w
            temp(i,j)=bitget(I(i,j),k);
        end
    end
subplot(3,3,9-k+1);imshow(temp,[ ]);
ind=num2str(k-1);
imti=['第',ind,'个位平面'];
title(imti);
end

level1 = 256;
level2 = 8;
ratio = level1/level2;
I1 = imread('cameraman.jpg');
subplot(121);imshow(I1);
S = size(I1);
for m = 1: S(1)
for n = 1: S(2)
I2(m,n) = uint8(round(double(I1(m,n))/ratio));
I2(m,n) = uint8(ratio*double(I2(m,n)));
end
end
subplot(122);imshow(I2);

F = imread('lena.bmp');
figure(1);imshow(F);
DCTF=dct2(F);
figure(2);imshow(log(abs(DCTF)),[ ]); %w´DCTXÍ„î
T1 = 5; T2 = 50;
DCT F(abs(DCT F) < T1) = 0;
IDCTF1=idct2(DCTF);
figure(3);imshow(IDCTF1,[0 255]);
DCT F(abs(DCT F) < T2) = 0;
IDCTF2=idct2(DCTF);
figure(4);imshow(IDCTF2,[0 255]);