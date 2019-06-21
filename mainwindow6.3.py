import sys
import cv2
from PyQt5.QtWidgets import *
from PyQt5 import QtGui,QtCore,QtWidgets
from PyQt5.QtGui import *
import tensorflow as tf
from PIL import Image, ImageFilter
import os
import numpy as np
import glob
import math

class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.mainwindow()


    def mainwindow(self):
        
        #主窗口大小和位置
        self.resize(800,600)
        self.move(300,100)
        self.setWindowTitle("试卷成绩采集系统")

        #按钮提示信息字体及字号
        QToolTip.setFont(QtGui.QFont("SansSerif",10))
        
        #识别按钮
        recognise_btn = QPushButton("打开本地图片",self)
        recognise_btn.setToolTip("点击此按钮，进行试卷成绩识别")
        recognise_btn.resize(110,40)
        recognise_btn.setFont(QtGui.QFont("SansSerif",10,QtGui.QFont.Bold))
        recognise_btn.move(650,150)
        recognise_btn.show()

         
        #拍照按钮
        takephoto_btn = QPushButton("打开摄像头",self)
        takephoto_btn.setToolTip("点击此按钮，进行试卷成绩识别")
        takephoto_btn.resize(110,40)
        takephoto_btn.setFont(QtGui.QFont("SansSerif",10,QtGui.QFont.Bold))
        takephoto_btn.move(650,240)
        takephoto_btn.show()
       
       

        #预处理按钮
        pre_btn = QPushButton("处理分析并计算",self)
        pre_btn.setToolTip("点击此按钮，进行试卷成绩总分计算")
        pre_btn.resize(110,40)
        pre_btn.setFont(QtGui.QFont("SansSerif",10,QtGui.QFont.Bold))

        pre_btn.move(650,330)
        pre_btn.show()

        #计算成绩按钮
        #save_btn = QPushButton("保存到excel",self)
        #save_btn.setToolTip("点击此按钮，进行试卷成绩保存到excel")
        #save_btn.resize(110,45)
        #save_btn.setFont(QtGui.QFont("SansSerif",13,QtGui.QFont.Bold))
        #save_btn.move(650,330)
        #save_btn.show()

        #打开本地图片。 ——————后期改为调用摄像头
        recognise_btn.clicked.connect(self.openimage)
        
        takephoto_btn.clicked.connect(self.takephoto)
        
        pre_btn.clicked.connect(self.pre)
        

        #各题成绩读取显示文本框label
        textlabel1 = QLabel(self)
        textlabel1.setFont(QtGui.QFont("SansSerif",15,QtGui.QFont.Bold))
        #QtCore.QRect(x,y,宽，高)
        textlabel1.setGeometry(QtCore.QRect(50, 450, 120, 30))
        textlabel1.setText("一：")
        self.text1 = QtWidgets.QLineEdit(self)
        self.text1.setGeometry(QtCore.QRect(80, 450, 70, 30))


        textlabel2 = QLabel(self)
        textlabel2.setFont(QtGui.QFont("SansSerif",15,QtGui.QFont.Bold))
        textlabel2.setGeometry(QtCore.QRect(200, 450, 120, 30))
        textlabel2.setText("二：")
        self.text2 = QtWidgets.QLineEdit(self)
   
        self.text2.setGeometry(QtCore.QRect(230, 450, 70, 30))

        textlabel3 = QLabel(self)
        textlabel3.setFont(QtGui.QFont("SansSerif",15,QtGui.QFont.Bold))
        textlabel3.setGeometry(QtCore.QRect(350, 450, 120, 30))
        textlabel3.setText("三：")
        self.text3 = QtWidgets.QLineEdit(self)
        self.text3.setGeometry(QtCore.QRect(380, 450, 70, 30))

        textlabel4 = QLabel(self)
        textlabel4.setFont(QtGui.QFont("SansSerif",15,QtGui.QFont.Bold))
        textlabel4.setGeometry(QtCore.QRect(500, 450, 120, 30))
        textlabel4.setText("四：")
        self.text4 = QtWidgets.QLineEdit(self)
        self.text4.setGeometry(QtCore.QRect(530, 450, 70, 30))

        textlabel5 = QLabel(self)
        textlabel5.setFont(QtGui.QFont("SansSerif",15,QtGui.QFont.Bold))
        textlabel5.setGeometry(QtCore.QRect(50, 500, 120, 30))
        textlabel5.setText("五：")
        self.text5 = QtWidgets.QLineEdit(self)
        self.text5.setGeometry(QtCore.QRect(80, 500, 70, 30))

        textlabel6 = QLabel(self)
        textlabel6.setFont(QtGui.QFont("SansSerif",15,QtGui.QFont.Bold))
        textlabel6.setGeometry(QtCore.QRect(200, 500, 120, 30))
        textlabel6.setText("六：")
        self.text6 = QtWidgets.QLineEdit(self)
        self.text6.setGeometry(QtCore.QRect(230, 500, 70, 30))

        textlabel7 = QLabel(self)
        textlabel7.setFont(QtGui.QFont("SansSerif",15,QtGui.QFont.Bold))
        textlabel7.setGeometry(QtCore.QRect(350, 500, 120, 30))
        textlabel7.setText("七：")
        self.text7 = QtWidgets.QLineEdit(self)
        self.text7.setGeometry(QtCore.QRect(380, 500, 70, 30))

        textlabel8 = QLabel(self)
        textlabel8.setFont(QtGui.QFont("SansSerif",15,QtGui.QFont.Bold))
        textlabel8.setGeometry(QtCore.QRect(500, 500, 120, 30))
        textlabel8.setText("八：")
        self.text8 = QtWidgets.QLineEdit(self)
        self.text8.setGeometry(QtCore.QRect(530, 500, 70, 30))


        textlabel9 = QLabel(self)
        textlabel9.setFont(QtGui.QFont("SansSerif",15,QtGui.QFont.Bold))
        textlabel9.setGeometry(QtCore.QRect(620, 470, 120, 50))
        textlabel9.setText("总分：")
        self.text9 = QtWidgets.QLineEdit(self)
        self.text9.setGeometry(QtCore.QRect(680, 470, 80, 60))



        
    def openimage(self):
        #图像显示label
        label = QLabel(self)
        label.setFixedSize(580,400)
        label.move(30,30)
        label.setStyleSheet("QLabel{background:white;}")
        
        print("load--file")
        # QFileDialog就是系统对话框的那个类第一个参数是上下文，第二个参数是弹框的名字，第三个参数是开始打开的路径，第四个参数是需要的格式
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        jpg = QtGui.QPixmap(imgName).scaled(label.width(), label.height())
        label.setPixmap(jpg)
        label.show()
       
    def takephoto(self):
        
        
        cap = cv2.VideoCapture(0)#启动摄像头，电脑摄像头默认为编号0
        #导入本地的人脸识别xml文件进行当前摄像头画面的人脸识别

        while(True):
            ret,label = cap.read()
            height, width, bytesPerComponent = label.shape
            bytesPerLine = bytesPerComponent * width
        
            hsv = cv2.cvtColor(label,cv2.COLOR_BGR2HSV)
            cv2.imshow('label',label)
            if cv2.waitKey(1) & 0xFF ==ord('t'): #设置按t进行拍照
                cv2.imwrite("./li2.jpg",label)
                break
            if cv2.waitKey(1) & 0xFF ==ord('q'): #设置长按q退出
                break
            #image = QtGui.QImage(label.data, width, height, bytesPerLine, QImage.Format_RGB888)
            #label.setPixmap(QtGui.QPixmap.fromImage(label).scaled(label.width(), label.height()))

        cap.release()#释放摄像头
        cv2.destroyAllWindows()#关闭当前摄像头的所有页面


        #图像显示label
        label = QLabel(self)
        label.setFixedSize(580,400)
        label.move(30,30)
        label.setStyleSheet("QLabel{background:white;}")
        pixmap = QPixmap ("./li2.jpg")  # 按指定路径找到图片，注意路径必须用双引号包围，不能用单引号
        label.setPixmap (pixmap)  # 在label上显示图片
        label.setScaledContents (True)  # 让图片自适应label大小


        label.show()

        




    def pre_test(self):
    # 读入原图片
        path_file_number=glob.glob('./cut_image/*.jpg') #获取当前文件夹下个数
        count=(len(path_file_number))
        for k in range(0,count):
            img = cv2.imread('./cut_image/%d.jpg' %k)

            # 将图片高和宽分别赋值给x，y
            x, y = img.shape[0:2]
            # 改变大小
            img_1 = cv2.resize(img, (28, 28),interpolation=cv2.INTER_CUBIC)
           # 变灰度图
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            # 二值化
            retval, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

           
            #print(dst.shape)
            save = cv2.imwrite("./cut_image/%d_.jpg" %k,dst)

        score_Arr = get_score()
        #print(score_Arr)
        self.text1.setText(str(score_Arr[0]))
        self.text2.setText(str(score_Arr[1]))
        self.text3.setText(str(score_Arr[2]))
        self.text4.setText(str(score_Arr[3]))
        self.text5.setText(str(score_Arr[4]))
        self.text6.setText(str(score_Arr[5]))
        self.text7.setText(str(score_Arr[6]))
        self.text8.setText(str(score_Arr[7]))
        print("22222")
        self.text9.setText("%d" %sum(score_Arr))


        






    #图像预处理
    def pre(self):
        Img = cv2.imread("./li.jpg")
        
        HSV = cv2.cvtColor(Img,cv2.COLOR_BGR2HSV)

        lower = np.array([30,30,90])
        upper = np.array([190,210,190])

        mask =cv2.inRange(HSV,lower,upper)
        print(mask)
        h = cv2.imwrite("./mask.jpg",mask)

        #自适应阈值二值化
        #dst = cv2.adaptiveThreshold(src, maxval, thresh_type, type, Block Size, C)
        #src： 输入图，只能输入单通道图像，通常来说为灰度图 
        #maxval： 当像素值超过了阈值（或者小于阈值，根据type来决定），所赋予的值
        #thresh_type： 阈值的计算方法，包含以下2种类型：cv2.ADAPTIVE_THRESH_MEAN_C； cv2.ADAPTIVE_THRESH_GAUSSIAN_C.
        #type：二值化操作的类型，与固定阈值函数相同，包含以下5种类型： cv2.THRESH_BINARY； cv2.THRESH_BINARY_INV； cv2.THRESH_TRUNC； cv2.THRESH_TOZERO；cv2.THRESH_TOZERO_INV.
        #Block Size： 图片中分块的大小
        #C ：阈值计算方法中的常数项

        binary = cv2.adaptiveThreshold(mask,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,4)
        binary_save = cv2.imwrite("./binary.jpg",binary)
        #cv2.getStructuringElement(内核形状（矩形：MORPH_RECT;交叉形：MORPH_CROSS;椭圆形：MORPH_ELLIPSE）,（内核的尺寸以及锚点的位置）)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

        #morphologyEx(
         #开运算(open)：先腐蚀后膨胀的过程。开运算可以用来消除小黑点，在纤细点处分离物体、平滑较大物体的边界的   同时并不明显改变其面积。
         #闭运算(close)：先膨胀后腐蚀的过程。闭运算可以用来排除小黑洞。
         #形态学梯度(morph-grad)：可以突出团块(blob)的边缘，保留物体的边缘轮廓。
         #顶帽(top-hat)：将突出比原轮廓亮的部分。
         #黑帽(black-hat)：将突出比原轮廓暗的部分。)
        closing = cv2.morphologyEx(binary,cv2.MORPH_CLOSE,kernel)
        #print(closing)

        #closing_save = cv2.imwrite("./mask1.jpg",closing)
        #cv2.imshow('closing', closing)


        #第一个参数是寻找轮廓的图像；
        #第二个参数表示轮廓的检索模式：cv2.RETR_EXTERNAL表示只检测外轮廓  cv2.RETR_LIST检测的轮廓不建立等级关系
        #cv2.RETR_CCOMP建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一个连通物体，这个物体的边界也在顶层。
        #cv2.RETR_TREE建立一个等级树结构的轮廓。

        #第三个参数method为轮廓的近似办法
        #cv2.CHAIN_APPROX_NONE存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max（abs（x1-x2），abs（y2-y1））==1
        #cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息
        #cv2.CHAIN_APPROX_TC89_L1，CV_CHAIN_APPROX_TC89_KCOS使用teh-Chinl chain 近似算法
        binary ,contours ,hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        print (len(contours))
        m = len(contours)


        # 排序

        px = -1
        swapped=True

        while swapped:
            swapped = False
            px = px + 1
            for i in range(1,m-px):
                x, y, w, h = cv2.boundingRect(contours[i-1])
                x2, y2, w2, h2 = cv2.boundingRect(contours[i])
                if x>x2:
                    temp = contours[i]
                    contours[i] = contours[i-1]
                    contours[i-1] = temp
                    swapped = True


        #识别框
        for i in range(0,len(contours)): 
            x, y, w, h = cv2.boundingRect(contours[i])  
            cv2.rectangle(Img, (x,y), (x+w,y+h), (255,255,0), 2)
            #图像分割
            newimage=Img[y+2:y+h-2,x+2:x+w-2] # 先用y确定高，再用x确定宽
            rootdir=("./cut_image/")
            if not os.path.isdir(rootdir):
              os.makedirs(rootdir)
            cv2.imwrite( rootdir+str(i)+".jpg",newimage) 
            print (i)

        #cv2.imshow("img", Img)
        cv2.imwrite("./shibie.jpg",Img)
       
        

        #图像显示label
        label = QLabel(self)
        label.setFixedSize(580,400)
        label.move(30,30)
        label.setStyleSheet("QLabel{background:white;}")
        
        print("load--file2")
        pixmap = QPixmap ("./shibie.jpg")  # 按指定路径找到图片，注意路径必须用双引号包围，不能用单引号
        label.setPixmap (pixmap)  # 在label上显示图片
        label.setScaledContents (True)  # 让图片自适应label大小


        label.show()

        self.pre_test()

def imageprepare(Img_Name):
    Img_Path = './cut_image/%s_.jpg'%Img_Name
    im = Image.open(Img_Path) #读取的图片所在路径，注意是28*28像素

    im = im.convert('L')
    tv = list(im.getdata()) 
    tva = [(255-x)*1.0/255.0 for x in tv] 
    return tva

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def file_name(file_dir):   
    for root, dirs, files in os.walk(file_dir):  
        return int(len(files)/2)


def get_score():
    rlist = []
    file_sum = file_name('./cut_image/')
    for i in range(0,file_sum):
        print(i)
        result=imageprepare(str(i))
        tf.reset_default_graph()
        x = tf.placeholder(tf.float32, [None, 784])

        y_ = tf.placeholder(tf.float32, [None, 10])


        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])

        x_image = tf.reshape(x,[-1,28,28,1])

        h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        keep_prob = tf.placeholder("float")
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        saver = tf.train.Saver()


        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, "./model/model.ckpt") #使用模型，参数和之前的代码保持一致
            
            prediction=tf.argmax(y_conv,1)
            predint=prediction.eval(feed_dict={x: [result],keep_prob: 1.0}, session=sess)

            #print('识别结果:')
            #print(predint[0])
            rlist.append(predint[0])
    return rlist



    
#    def CNN(self):
        
    

if __name__=='__main__':
    app  = QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())

    

