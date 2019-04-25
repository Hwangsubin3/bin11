#시즌2 Lab-05 Logistic Regression

import tensorflow.contrib.eager as tfe  # 텐서플로우에서 eager모드로 실행하기 위해 import한다.
tf.enable_eager_execution()   #eager_execution을 선언한다.
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(len(x_train)) #tf데이터를 통해서 x값과 y값을 실제 x의 길이만큼 
                                                                                      #batch시켜서 학습한다는 것을 dataset에 저장한다.
W = tf.Variable(tf.zeros([2,1]), name='weight') # 2행1열의 모양으로 w값을 선언해준다.
b = tf.Variable(tf.zeros([1]), name='bias')     #b(bias)값도 선언해준다.

def logistic_regression(features):
    hypothesis  = tf.div(1., 1. + tf.exp(tf.matmul(features, W) + b)) #linear한값, wx+b를 sigmoid한 값으로 hypothesis를 지정해준다.
    return hypothesis        #logistic_regression에 관한 hypothesis를 그려낼 수 있다.

 def loss_fn(features, labels):
    hypothesis = logistic_regression(features)  #위에서 나온 hypothesis값에 labels를 적용한다.
    cost = -tf.reduce_mean(labels * tf.log(logistic_regression(features)) + (1 - labels) * tf.log(1 - hypothesis))
    return cost   #lables과 hypothesis값들을 통해서 우리가 원하는 cost값을 구한다.

 def grad(hypothesis, features, labels):  #hypothesis값과 labels값을 불러온다.
    with tf.GradientTape() as tape:
        loss_value = loss_fn(hypothesis,labels)  #hypothesis(가설값)과 lables(실제값)을 비교한 loss값을 구한다.
    return tape.gradient(loss_value, [W,b])  #gradient를 통해서 실제 모델값을 계속 변화시킨다.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)  #GradientDescentOptimizer를 통해서 실제 우리가 이동할 learning_rate 값을
                                                                   #통해 optimizer를 선언한다.
for step in range(EPOCHS):   #함수를 EPOCHS만큼 반복한다.
    for features, labels  in tfe.Iterator(dataset):  #dataset을 Iterator로 돌려서 x,y값을 넣어가면서 반복.    
        grads = grad(logistic_regression(features), features, labels)  #나온x,y값을 가설에 대입해서 grads값을 불러온다.
        optimizer.apply_gradients(grads_and_vars=zip(grads,[W,b]))   #gradients과정을 통해서 계속해서 w,b값이 업데이트된다.
        if step % 100 == 0:   #100번마다 학습
            print("Iter: {}, Loss: {:.4f}".format(step, loss_fn(logistic_regression(features),features,labels))) #Iter,Loss값 출력

 def accuracy_fn(hypothesis, labels):  #가설과 함수를 비교하기 위해 accuracy_function을 이용한다.
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)  #hypothesis값이 0.5보다 큰지 predicted값을 불러온다.
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, labels), dtype=tf.int32))  #위에서의 예측된 값이 실제 나온값과 비교해서 맞는지
    return accuracy                                                                  #accuracy에 출력한다.
            
test_acc = accuracy_fn(logistic_regression(x_test),y_test)  #x_test,y_test값을 test_acc에 넣어 값을 출력한다.

