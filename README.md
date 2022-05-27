# OpenCvRoute
opencv学习路径 

# OpenCv能做什么
+ 目标识别
  + 人脸识别
  + 面部效果
  + 车辆识别
  + 物品分类
+ 自动驾驶
+ 医学图像分析
+ 视频内容理解
  + 视频中文字的识别
  + 公众人物的识别
  + 各种物品的识别
  + 人物出现的时间关系

#Api使用
+ 创建和显示窗口
  + namedWindow()
  + imshow() 
  + destoryAllWindows()
  + resizeWindow
+ 加载显示图片
  + imread
  + imshow
+ 保存图片
  + imwrite(name,img)
  + name 要保存的文件名
  + img, 是Mat的类型
+ 视频采集
  + VidwoCapure() 视频采集
  + cap.read() 读取视频
  + cap.release() 释放视频资源
  + read 
   + 返回两个值，第一个为状态值，读到帧为true
   + 第二个值为视频帧
+ 视频录制
  + VideoWriter 参数一位输出文件，参数二位多媒体文件格式，参数三为帧率，参数四为分辨率大小
    + write
    + release
+ 设置鼠标回调函数
  + setMouseCallback(winname,callback,userdata)
  + callback(event,x,y,flags,userdata)
    + event:鼠标移动、按下左键
    + x.y:鼠标坐标
    + flags: 鼠标键及组合键
+ TrackBar控件
  + createTrackBar
    + tranckbarname 、winname 
    + value : trackbar 当前的值
    + count ,最小值为0 最大值为count
    + callback,userdata
  + getTrackBars
    + 输入参数：trackbarname 
    + 输入参数：winname
    + 输出：当前值
+ 色彩空间变换
  + RGB：人眼的色彩空间
  + OpenCV默认使用BGR
  + HSV/HSB/HSL 色相/明亮度/
    + HSV
      + Hue:色相，及色彩，如红色、蓝色
      + Saturation: 饱和度，颜色的纯度
      + Value:明度 
    + HSL
      + Hue:色相，及色彩，如红色、蓝色
      + Saturation: 饱和度，颜色的纯度
      + Light:亮度 
  + YUV 视频领域
    + YUV 4:2:0
    + YUV 4:2:2
    + YUV 4:4:4
  + Numpy 
    + OpenCV中用到的矩阵都要转换成Numpy数组
    + Numpy是一个经高度优化的Python数值库
    + 创建矩阵
      + 创建数组 array()
      + 创建全0数组 zeros/ones
      + 创建全值数组 full()
      + 创建单元数组 identify/eye()
    + 检索与赋值
    + 获取子数组
+ 像素访问
  + Mat实现: Header(头部)、Data(数据) 
  + Mat数据结构
    + int dims: 维数
    + int rows,cols; 行列数
    + uchar *data: 存储数据的指针
    + int *refcount：引用计数
    + depth: 像素的位深
    + channels：通道数 RGB是3
    + size: 矩阵大小
    + type: dep+dt+chs CV_8UC3
    + data: 存放数据
  + Mat深拷贝和浅拷贝
    + Mat共享数据
    + Mat浅拷贝
      + Mat A  A=cv2.imread(file,IMREAD_COLOR) MAT B(A)
      + cv::Mat::clone 深拷贝
      + cv::Mat::copyTo 深拷贝
      + copy() 深拷贝 
  + 访问Mat属性
    + shape、size、dtype
  + 通道的分离与合并
    + split(mat)
    + merge((ch1,ch2,...))
+ 基本图形的绘制
  + 画线
    + line(img,开始点,结束点，颜色)
    + img: 在哪个图像上画线
    + 开始点、结束点： 指定线的开始与结束位置
    + 颜色、线宽、线型
  + 画椭圆
    + ellipse(img,中心点，长宽的一半，角度，从哪个角度开始，从哪个角度结束)
  + 画多边形
    + polylines(img,点集，是否闭环,颜色,.)
  + 画文本
    + putText(img,字符串,起始点，字体，字号)
+ 图像的运算
  + 图像相加 add 
  + 图像相减 subtract 
  + 图像乘与除 multiply divide 
  + 图像融合 addWeighted(A,alpha,B,bate,gamma)
    + alpha 和 beta是权重 
    + gamm 是静态权重 
  + 图像位运算
    + 与
    + 或
    + 非
    + 异或
+ 图像的变换
  + 图像的缩放
    + resize(src,dst,dsize,fx,fy,interpolation)
      + interpolation 缩放算法
        + INNER_NEAREST: 邻近插值，速度快，效果差
        + INNER_LINEAR: 双线性插值，院途中的4个点
        + INNER_ CUBIC:三次插值，原图中的16个点
        + INNER_AREA:效果最好
  + 图像翻转
    + flip(img,flipCode) 
      + flipcode == 0 上下
      + flipcode >0 左右
      + flipcode<0 上下 + 左右 
  + 图像旋转
    + rotate(img,rotateCode)
      + ROTATE_90_CLOCKWISE
      + ROTATE_180
      + ROTATE_90_COUNTERCLOCKWISE
  + 图像的仿射变换(旋转，缩放，平移的总称)
    + 仿射API
      + warpAffine(src,M,dsize,flags,mode,value)
      + M:变换矩阵
      + dsize: 变换后的大小
      + flags：与resize中的插值算法一致
      + mode: 边界外推法标志
      + value：填充边界的值
    + 平移矩阵
      + 矩阵中的每个像素由(x,y)组成
      + 因此，其变换矩阵是2×2的矩阵
      + 平移向量为2×1的向量，所在的平移矩阵为2×3矩阵
      + 变换矩阵
        + getRotationMatrix2D(center,angle,scale)
        + center 中心点
        + angle 角度
        + scale 缩放比例
    + 透视变换API
      + warpPerspective(img,M,dsize)
      + M 是变换矩阵
      + dszie是目标图像大小
      + getPersectiveTransform(src,dst)
      + 四个点(图形的四个角)
+ 图像滤波
  + 滤波的作用
    + 一副图像通过滤波器得到另一幅图像
      + 其中滤波器又称为卷积核，滤波的过程称为卷积  
  + 卷积的基本概念
    + 卷积核的大小
      + 卷积核一般为奇数，如3*3 5*5 7*7
        + 一方面是增加padding的原因
        + 另一方面是保证锚点在中间，防止位置发生偏移的原因
      + 卷积核的大小的影响
        + 在深度学习中，卷积核越大
        + 看到的信息(感受野)越多
        + 提取的特征越好，同时计算量也就越大
    + 锚点(中心点)
    + 边界扩充
      + 当卷积核大于1且不进行边界扩充，输出尺寸将相应缩小
      + 当卷积核以标准方式进行边界扩充，则输出数据的空间尺寸将于输入相等
      + 计算公式
        + N = （W - F +2P)/S +1
        + N 输出图像大小
        + W 源图大小；F 卷积核大小；P 扩充尺寸
        + S 步长大小 
    + 步长(间隔)
  + 低通滤波、高通滤波
    + 低通滤波可以去除噪音或平滑图像
    + 高通滤波可以帮助查找图像的边缘 
  + 图像卷积
    + filter2D(src,ddepth,kernel,anchor,delta,borderType)
      + ddepth 位深 默认为-1
      + kernal 卷积核
      + anchor 锚点
      + delta 
      + borderType 边界类型
  + 方盒滤波与均值滤波
    + 方盒滤波卷积核 
      + 参数a的作用    
        + normalize = true, a = 1/W* H  方盒滤波 == 均值滤波 
        + normalize = false, a = 1 
      + boxFilter(src,depth,ksize,anchor,normalize,borderType)
    + 均值滤波
      + blur(src,ksize,anchor,borderType)  
  + 高斯滤波。对高斯噪点有效果
    + GasussianBlur(img,kernal,sigmaX,sigmaY)
  + 中值滤波。取其中的中间值作为卷积后的结果值。对胡椒噪音效果明显
    + medianBlur(img,ksize)
  + 双边滤波
    + 优点
      + 可以保留边缘
      + 同时可以对边缘内的区域进行平滑处理 
    + 作用：进行美颜 
    + API
      + bilateralFilter(img,d,sigmaColor,sigmaSpace)
  + 高通滤波
    + Sobel(索贝尔)(高斯)
      + 先向X方向求导
      + 再向y方向求导
      + 最终结果：|G| = |Gx| + |Gy| 
      + Sobel(src,ddepth,dx,dy,ksize=3)
    + Scharr(沙尔) 卷积核不可改变。只能求一个方向。
      + 与Sobel类似，只不过是用的kernel值不同
      + 只能求x方向或y方向的边缘
    + Laplacian(拉普拉斯)
      + 可以同时求两个方向的边缘
      + 对噪音敏感，一般需要先进行去噪再调用拉普拉斯
    + Canny(求边缘)
      + 使用5×5高斯滤波消除噪声
      + 计算图像梯度的方向（0/45/90/135）
      + 取局部极大值
      + Canny(img,minVal,maxVal)
+ 形态学 
  + 基于图像形态进行处理的一些基本方法
  + 这些处理方法基本是对二进制图像（黑白）进行处理
  + 卷积核决定着图像处理后的效果
  + 形态学图像处理
    + 腐蚀与膨胀 
    + 开运算
    + 闭运算
    + 顶帽
    + 黑帽
  + 图像的二值化
    + 将图像的每个像素变成两种值，如0，255
    + 全局二值化
    + 局部二值化
    + threshold(img,thresh,maxVal,type)
      + img:图像，最好是灰度图。
      + thresh:阈值
      + maxVal: 超于阈值，替换成maxVal
      + Thresh_binary 
      + Thresh_binaer_inv
      + Thresh_trunc
      + Thresh_tozero 和 thresh_tozero_inv
    + 自定义阈值
      + adaptiveThreshold(img,maxVal,adaptiveMethodtype,blockSize,c)
        + adaptiveMethod:计算阈值的方法 
        + blockSize：邻近区域的大小
        + C ：常量，应从计算出的平均值和加权平均值中减去
        + APAPTIVE_THRESH_MEAN_C: 计算临近区域的平均值
        + ADAPTIVE_THRESH_GAUSSIAN_C:高斯窗口加权平均值
        + type： THRESH_BINARY、THRESH_BINARY_INV
  + 腐蚀的原理
    + erode(img,kernel,iterations=1)
    + 获取卷积核
      + getStructuringElement(type,size)
      + size值为：(3,3)、(5,5)
      + MORPH_RECT
      + MORPH_ELLIPSE
      + MORPH_CROSS
  + 膨胀
    + dilate(img,kernel,iterations = 1)
  + 开运算 = 腐蚀 + 膨胀 
    + morphologyEx(img,MORPH_OPEN,kernel)
  + 闭运算 = 膨胀 + 腐蚀
  + 形态学梯度 (求边缘)
    + 梯度 = 原图 - 腐蚀 
  + 顶帽运算
    + 顶帽 = 原图 - 开运算
  + 黑帽运算
    + 黑帽 = 原图 - 闭运算
  + 开运算，先腐蚀后膨胀，去除大图形外的小图形
  + 闭运算，先膨胀后腐蚀，去除大图行内的小图形
  + 梯度，求图形的边缘
  + 顶帽，原图减去开运算，得到大图形外的小图形
  + 黑帽，原图减去闭运算，得到大图形内的小图形 
+ 图像轮廓
  + 具有相同颜色或强度的连续点的曲线
  + 作用
    + 用于图形分析
    + 物体的识别与检测
  + 注意点
    + 为了检测的准确性，需要对图像进行二值化或Canny操作
    + 画轮廓时会修改输入的图像
    + findContours(img,mode,ApproxinmationMode)
      + 两个返回值，contours和hierarchy
      + mode
        + RETR_EXTERNAL = 0 表示只检测外轮廓
        + RETR_LIST = 1 ,检测的轮廓不建立等级关系
        + RETR_CCOMP = 2 ,每层最多两级
        + RETR_TREE = 3，按树形存储轮廓
      + ApproxinmationMode
        + CHAIN_APPROX_NONE, 保存所有轮廓上的点
        + CHAIN_APPROX_SIMPLE，只保存角点
  + 绘制轮廓
    + drawContours(img,contours,contourIndex,color,thickness...)
      + coutourIdx, -1 表示绘制所有轮廓
      + color,颜色(0,0,255)
      + thickness,线宽 -1 是全部填充
  + 轮廓面积
    + contourArea(countour)
    + contour:轮廓   
  + 轮廓周长
    + arcLength(curve,closed)
    + curve：轮廓
    + closed：是否是闭合的轮廓
  + 多边形逼近与凸包
    + 多边形逼近
      + approxPolyDP(curve,epsilon,closed)
      + curve：轮廓
      + eplison :精度
      + closed: 是否闭合 
    + 凸包
      + convexHull(points,clockwise)
      + points:轮廓
      + clockwise:顺时针绘制
  + 外接矩形
    + 最小外接矩形
      + minAreaRect(point)
        + point 轮廓
        + 返回值： RotateRect
          + x,y
          + width,height
          + angle 
    + 最大外接矩形
      + boundingRect(array)
        + array:轮廓
        + 返回 rect
+ Opencv特征的场景
  + 图像搜索，如以图搜图
  + 拼图游戏
    + 拼图方法
      + 寻找特征
      + 特征是唯一的
      + 能比较的 
      + 可追踪的
    + 平坦部分很难找到它在原图中的位置
    + 边缘相比平坦更好找一些，但也不能一下确定
    + 角点可以一下就能找到其在原图的位置
    + 特征
      + 图像特征就是指有意义的图像区域，具有独特性，易于识别性，比如角点、斑点及高密度区
      + 角点
        + 灰度梯度的最大值对应的像素
        + 在特征中最重要的是角点
        + 两条线的交点
        + 极值点（一阶导数最大值，但二阶导数为0）
      + Harris角点
        + 光滑地区，无论向哪里移动，衡量系数不变
        + 边缘地址，垂直边缘移动时，衡量系统变化剧烈
        + 在交点处，往哪个方向移动，衡量系统都变化剧烈
        + blockSize ：检测窗口大小
        + ksize：Soble的卷积核
        + k：权重系数，经验值，一般取0.02-0.04之间
      + Shi-Tomasi角点检测
        + Shi-Tomasi是Harris角点检测的改进
        + Harris角点检测算得稳定性和K有关，而K是个经验值，不好设定最佳值
        + goodFeaturstoTrack
          + maxCorners:角点的最大数，值为0表示无限制
          + qualityLevel:小于1.0的正数，一般在0.01-0.1之间
          + minDistance: 角之间最小欧式距离，忽略小于此距离的点
          + mask ：感兴趣的区域
          + blocksize: 检测窗口
          + useHarrisDetector: 是否使用Harris算法
          + k 默认是0.04
      + SIFT
        + 出现的原因
          + Harris角点具有旋转不变的特性 
          + 但缩放后，原来的角点有可能就不是角点了。
        + 创建SIFT对象
          + 进行检测 kp = sift.detect(img,...)
          + 绘制关键点，drawKeyPoints(gray,kp,img)
        + 关键点和描述子
          + 关键点：位置、大小和方向
          + 描述子：记录了关键点周围对其有贡献的像素点的一组向量值，其不受仿射变换，光照变换等影响
          + 计算描述子
            + sift.compute(img,kp)
            + 其作用是进行特征匹配
        + 同时计算关键点和描述子
          + kp,des = sift.detactAndCompute(img,..)
          + mask 指明对img中哪个区域进行计算
      + SURF
        + 优点：速度快
        + SIFT：最大的问题是速度慢  
      + ORB 
        + ORB可以做到实时监测
    + 特征匹配方法
      + BF 暴力特征匹配方法
        + 它使用第一组中的每个特征的描述子 与第二组中的所有特征描述子进行匹配
        + 计算它们之间的差距，然后将最接近一个匹配返回
        + 特征匹配步骤
          + 创建匹配器，BFMatcher
            + normType: NORM_L1,NORM_L2,HAMMING1
            + crossCheck:是否进行交叉匹配，默认为false
          + 进行特征匹配，bf.match(des1,des2)  
            + 参数为SIFT,SURF,OBR等算子
          + 绘制匹配点，cv2.drawMatches(img1,kp1,img2,kp2)
            + 搜索img,kp
            + 匹配图img,kp
            + match()方法返回的匹配结果
      + FLANN 最快邻近区特征匹配方法 
        + 在进行批量特征匹配时，Flann速度更快
        + 由于它使用的是邻近近似值，所以精度较差
        + 创建匹配器，FlannBasedMatcher
          + index_params 字典：匹配算法 KDTREE。LSH
          + search_params 字典: 指定KDTREE算法中遍历树的次数
            + KDTREE。 trees = 5 ,search_params 50 
          + knnMatch方法
            + 参数为SIFT、SURF、ORB等计算的描述子
            + k，表示取欧式距离最近的前K个关键点
            + 返回的是匹配的结果
          + trainIdx
        + 进行特征匹配，flann.match/knnMatch  
        + 绘制匹配点 cv2.drawMatcher/drawMatcherKnn
      + 图像查找
        + 单应性矩阵
      + 图像合并的步骤
        + 读文件并重置尺寸
        + 根据特征点和计算描述子，得到单应性矩阵
        + 图像变换     
        + 图像拼接并输出图像
  + 图像的分割
    + 将前景物体从背景中分离出来
    + 图像分割的方法
      + 传统的图像分割方法
        + 分水岭法。图像存在过多的极小区域而产生很多小的集水盆
          + 标记背景
          + 标记前景
          + 标记未知域
          + 进行分割
          + API
            + watershed(img,masker)
            + masker,前景，背景设置不同的值用以区分它们
        + GrabCut法
        + MeanShift法
        + 背景扣除
      + 基于深度学习的图像分割方法
  + 机器学习
    + 人脸识别
      + 哈尔(Haar)级联方法
        + 创建Haar级联器
        + 导入图片并将其灰度化
        + 调用detectMultiScale方法进行人脸识别
          + (img,double scaleFactor = 1.1,int minNeighbors = 3)
      + 深度学习方法(DNN)
        + OpenCV对DNN的支持
          + 只能使用DNN，不能训练DNN模型
          + 支持Tensflow/Pytorch/Caff/DarkNet
            + 读取模型，并得到深度神经网络
              + readNetFromTensflow(model,config)
              + readNetFromCaffe(config,model)
              + readNetDarknet(config,model),YOLO
              + readNet(model,[config,[framework]])
            + 读取图片/视频
            + 将图片转成张量，送入深度神经网络
              + blobFromImage 函数
                + image 
                + scalefactor = 1.0 缩放因子
                + size = Size()
                + mean = Scalar()
                + swapRB = false
                + crop = false 
            + 进行分析，并得到结果 
    + 车牌识别
      + Haar + Tesacct 识别车牌
      + 通过Haar 定位车牌的大体位置
      + 对车牌进行预处理
        + 对车牌进行二值化处理
        + 进行形态学处理
        + 滤波去除噪点
        + 缩放
      + 调用tesseract进行文字识别
      
      
  