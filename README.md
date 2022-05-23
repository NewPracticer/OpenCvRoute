# OpenCvRoute
opencv学习路径

# OpenCv学习路径
+ [入门OpenCv](https://coding.imooc.com/class/496.html)
  + 对应案例
    + 车辆检测、图像拼接、文字识别
+ [深入OpenCv](https://ke.qq.com/course/456184 )
  + 对应案例
    + [信用卡数字识别、文档扫描ORC识别、图像拼接、停车场车位识别、答题卡识别、目标追踪、疲劳检测](https://www.bilibili.com/video/BV1PV411774y?spm_id_from=333.337.search-card.all.click)
+ [OpenCv全流程学习](https://ke.qq.com/course/package/38162)
  + 对应案例
    +  刀片检测、二维码检测、人脸检测、车牌检测、行人检测、图像拼接
+ [工业OpenCv使用](https://www.kaikeba.com/course/vip/788)
  + 对应案例
    + 文档扫描OCR识别、零件识别、人脸识别饰品添加、目标追踪、目标检测 

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
    + 锚点
    + 边界扩充
      + 当卷积核大于1且不进行边界扩充，输出尺寸将相应缩小
      + 当卷积核以标准方式进行辩解扩充，则输出数据的空间尺寸将于输入相等
    + 步长
      