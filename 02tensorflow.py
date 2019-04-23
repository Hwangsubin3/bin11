#시즌2 Lab-02 Simple Linear Regression LAB
import tensorflow as tf  #tensorflow를 tf로 불러온다.
tf.enable_eager_execution() #즉시 실행(eager_execution)모드로 변경해서 사용한다.

# Data
x_data = [1, 2, 3, 4, 5]  # x데이터를 불러온다.(input)
y_data = [1, 2, 3, 4, 5]  # y데이터를 불러온다.(output)

# W, b initialize
W = tf.Variable(2.9)    # w에 임의의 값을 정한다.
b = tf.Variable(0.5)    # b에 임의의 값을 정한다.

learnig_rate=0.01  #learning_rate 학습률을 0.01로 줌.

# W, b update    
for i in range(100+1):   #학습을 101번 반복
    # Gradient descent     # 경사하강법을 이용해서 w와b값을 계속 변화시킨다.
    with tf.GradientTape() as tape:
        hypothesis = W * x_data + b
        cost = tf.reduce_mean(tf.square(hypothesis - y_data))  #cost값을 가설과 실제값y의 차이의 평균으로 지정한다.
    W_grad, b_grad = tape.gradient(cost, [W, b])   #cost함수에서 두개의 변수 w,b에 대한 미분값을 각각 나타낸다.
    W.assign_sub(learning_rate * W_grad)  #assign_sub를 이용해서 w값을 업데이트한다.
    b.assign_sub(learning_rate * b_grad)  #assign_sub를 이용해서 b값을 업데이트한다.
    if i % 10 == 0:   #i가 10의 배수가 될때마다 출력한다.
      print("{:5}|{:10.4f}|{:10.4f}|{:10.6f}".format(i, W.numpy(), b.numpy(), cost))

print()
