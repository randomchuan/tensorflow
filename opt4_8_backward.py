import tensorflow as tf
import opt4_8_forward as forward
import opt4_8_generateds
import numpy as np
import matplotlib.pyplot as plt

#常量定义
REGULARIZER = 0.01
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.999
BATCH_SIZE = 30
STEPS = 40000


def backward():
    #外标
    x = tf.placeholder(tf.float32,shape=(None,2))
    y_ = tf.placeholder(tf.float32,shape = (None,1))
    
    #数据
    X,Y_,Y_c = opt4_8_generateds.generateds()
    
    #复现网络结构，推测输出y
    y = forward.forward(x,REGULARIZER)
    global_step = tf.Variable(0,trainable = False)
    
    #定义损失,我用均方误差
    loss_mse = tf.reduce_mean(tf.square(y-y_))
    #加入正则化
    loss = loss_mse + tf.add_n(tf.get_collection('losses'))
    
    #我是用指数衰减学习率
    learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,global_step,
            300/BATCH_SIZE,LEARNING_RATE_DECAY,
            staircase = True
            )
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
            loss,global_step = global_step)
    
    #session结构初始化参数
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        for i in range(STEPS):
            start = (i*BATCH_SIZE)%300
            end = start + BATCH_SIZE
            sess.run(train_step,
                     feed_dict = {x:X[start:end],y_:Y_[start:end]})
            if i%2000 == 0:
                print("经过%d轮训练，损失为%f" %(i,sess.run(loss,feed_dict={x:X,y_:Y_})))
            
        #画出来
        #在-3到3之间以0.01为间距画网格
        xx,yy = np.mgrid[-3:3:.01,-3:3:0.01]
        grid = np.c_[xx.ravel(),yy.ravel()]
        probs = sess.run(y,feed_dict = {x:grid})
        probs = probs.reshape(xx.shape)
    #画点
    plt.scatter(X[:,0],X[:,1],c = np.squeeze(Y_c))
    plt.contour(xx,yy,probs,levels=[.5])
    plt.show()
    
    
    pass
    

if __name__ == "__main__":
    backward()