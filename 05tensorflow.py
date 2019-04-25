#Lab-05 Logistic Regression

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
                                                                   #통해
for step in range(EPOCHS):
    for features, labels  in tfe.Iterator(dataset):
        grads = grad(logistic_regression(features), features, labels)
        optimizer.apply_gradients(grads_and_vars=zip(grads,[W,b]))
        if step % 100 == 0:
            print("Iter: {}, Loss: {:.4f}".format(step, loss_fn(logistic_regression(features),features,labels)))
test_acc = accuracy_fn(logistic_regression(x_test),y_test)

print("Testset Accuracy: {:.4f}".format(test_acc))
