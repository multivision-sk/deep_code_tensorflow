import tensorflow as tf

#mnist 데이터 다운
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("tmp/data/",one_hot=True)

#학습을 위한 설정값 정의
learning_rate = 0.01
num_epochs = 5 #학습횟수
batch_size = 256 #배치 개수
display_step = 1 #손실함수 출력 주기
input_size = 784 #28x28 image data
hidden1_size = 256
hidden2_size = 256
output_size = 10

#입력값과 출력값을 받기위한 플레이스 홀더 정의
x = tf.placeholder(tf.float32,shape=[None,input_size])
y = tf.placeholder(tf.float32,shape=[None,output_size])

#ANN 모델 정의
def build_ANN(x):
    W1 = tf.Variable(tf.random_normal(shape=[input_size,hidden1_size]))
    b1 = tf.Variable(tf.random_normal(shape=[hidden1_size]))
    H1_output = tf.nn.relu(tf.matmul(x,W1)+b1)
    W2 = tf.Variable(tf.random_normal(shape=[hidden1_size,hidden2_size]))
    b2 = tf.Variable(tf.random_normal(shape=[hidden2_size]))
    H2_output = tf.nn.relu(tf.matmul(H1_output,W2)+b2)
    W_output = tf.Variable(tf.random_normal(shape=[hidden2_size,output_size]))
    b_output = tf.Variable(tf.random_normal(shape=[output_size]))
    logits = tf.matmul(H2_output,W_output)+b_output

    return logits

#ANN 모델 선언
predicted_value = build_ANN(x)

#손실함수와 옵티마이저 정의
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predicted_value, labels=y))
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

#세션을 열고 그래프 실행
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) # 변수들에 초기값 할당

    #지정된 횟수 만큼 최적화 수행
    for epoch in range(num_epochs):
        average_loss = 0.
        #전체 배치를 불러옴
        total_batch = int(mnist.train.num_examples/batch_size)
        #모든 배치들에 대해서 최적화 수행
        for i in range(total_batch):
            batch_x,batch_y = mnist.train.next_batch(batch_size)
            #옵티마이저하여 파라미터 업데이트
            _, current_loss = sess.run([train_step,loss], feed_dict={x:batch_x,y:batch_y})
            #평균 손실 측정
            average_loss += current_loss / total_batch

            #지정된 epoch 마다 학습결과 출력
            if epoch % display_step == 0 :
                print("반복(epoch):%d, 손실함수(loss):%f" %((epoch+1),average_loss))

            #테스트 데이터를 적용하여 학습 모델의 정확도 출력

            correct_prediction = tf.equal(tf.argmax(predicted_value,1),tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
            print("정확도:%f" %(accuracy.eval(feed_dict={x:mnist.test.images,y:mnist.test.labels})))

