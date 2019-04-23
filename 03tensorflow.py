#Lab-03 Linear Regression and Minimize cost 

import numpy as np   # numpy를 import한다.

X = np.array([1, 2, 3]) # x입력값을 지정한다.
Y = np.array([1, 2, 3]) # y출력값을 지정한다.

def cost_func(W, X, Y):  #데이터 x,y에 대해 w값이 주어졌을때 cost값을 계산하는 함수
    c = 0
    for i in range(len(X)):
        c += (W * X[i] - Y[i]) ** 2  #우리가 예측한값과 실제값의 차이를 제곱한값을 C에 누적한다.
    return c / len(X)                #누적한 값을 데이터의 개수로 나눠준다(len(x)). 즉 평균값

for feed_W in np.linspace(-3, 5, num=15): #-3과 5사이를 15번 구간으로 나눠 값을 가진다.
    curr_cost = cost_func(feed_W, X, Y)   #feed_w값에 따라 바뀌는 cost값의 변화를 저장한다.
    print("{:6.3f} | {:10.5f}".format(feed_W, curr_cost))
    
#Gradient descent
tf.set_random_seed(0)  #random_seed를 특정한 값으로 초기화한다.
x_data = [1., 2., 3., 4.]  # x데이터를 지정한다.
y_data = [1., 3., 5., 7.]  # y데이터를 지정한다.

W = tf.Variable(tf.random_normal([1], -100., 100.))  #random_normal(정규분포)를 따르는 1개짜리 변수를 만들어 w에 할당한다.

for step in range(300):  #gradient descent부분을 300번 반복한다.
    hypothesis = W * X   #hypothesis는 wx로 정의.
    cost = tf.reduce_mean(tf.square(hypothesis - Y))  #cost값을 차이의 제곱의 평균으로 지정한다.

    alpha = 0.01
    gradient = tf.reduce_mean(tf.multiply(tf.multiply(W, X) - Y, X))
    descent = W - tf.multiply(alpha, gradient)
    W.assign(descent)
    
    if step % 10 == 0:  #10번 반복 하면서 cost값과 w값을 출력한다.
        print('{:5} | {:10.4f} | {:10.6f}'.format(
            step, cost.numpy(), W.numpy()[0]))
