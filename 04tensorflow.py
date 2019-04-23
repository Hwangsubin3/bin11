시즌2  #Lab-04 Multi variable linear regression LAB

# data and label
x1 = [ 73.,  93.,  89.,  96.,  73.]  #입력 데이터값인 X1을 지정한다.
x2 = [ 80.,  88.,  91.,  98.,  66.]  #입력 데이터값인 X2을 지정한다.
x3 = [ 75.,  93.,  90., 100.,  70.]  #입력 데이터값인 X3을 지정한다.
Y  = [152., 185., 180., 196., 142.]  #츨력값 데이터(예측값)을 나타냄.

# random weights
w1 = tf.Variable(tf.random_normal([1])) #X변수도 3개이므로 Weight값도 3개로 지정한다.
w2 = tf.Variable(tf.random_normal([1])) #초기값은 모두 1로 준다.
w3 = tf.Variable(tf.random_normal([1]))
b  = tf.Variable(tf.random_normal([1])) #bias값은 1개로 지정한다.

learning_rate = 0.000001  #learning_rate값은 작은 수로 준다.

for i in range(1000+1):  #학습을 1001번 반복해준다.
    # tf.GradientTape() to record the gradient of the cost function
    with tf.GradientTape() as tape:  #GradientTape를 이용해 학습한다.
        hypothesis = w1 * x1 +  w2 * x2 + w3 * x3 + b  #
        cost = tf.reduce_mean(tf.square(hypothesis - Y)) #cost값은 가설값에서 실제값을 뺀것의 제곱의 평균으로 지정한다.
    # calculates the gradients of the cost
    w1_grad, w2_grad, w3_grad, b_grad = tape.gradient(cost, [w1, w2, w3, b])  #tape의 gradient값을 호출해서 cost함수에 대한 4개의 변수를 gradient값 기울기를 구한다.
    
    # update w1,w2,w3 and b
    w1.assign_sub(learning_rate * w1_grad)  #assign_sub를 이용해 w1~w3값을 업데이트한다.
    w2.assign_sub(learning_rate * w2_grad)
    w3.assign_sub(learning_rate * w3_grad)
    b.assign_sub(learning_rate * b_grad)   #assign_sub를 이용해 bias값도 업데이트 해준다.

    if i % 50 == 0:   #1001번 학습중 50번마다 cost값을 출력시킨다.
      print("{:5} | {:12.4f}".format(i, cost.numpy()))
