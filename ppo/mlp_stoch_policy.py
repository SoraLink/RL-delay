import tensorflow as tf, numpy as np

class Policy():
    def __init__(self,
                 sess,
                 state_dim,
                 action_dim,
                 name,
                 hidden_units,
                 ):
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.scope = name
        self.hidden_units = hidden_units
        self.state,self.mu, self.logstd, self.value = self.create_network()
        self.sigma = tf.exp(self.logstd)
        self.shape = tf.placeholder(shape=(2),dtype=tf.int32)
        self.x = self.mu + self.sigma * tf.random_normal(tf.shape(self.mu))
        self.action = self._produce_action(self.x)
        # self.mu = 0.1 * (tf.nn.tanh(self.mu)*0.5+0.5)+0.01 # shift the output to [0, 0.1]+0.01
        # self.sigma = 0.005 * self.sigma
        # x = self.mu + self.sigma * tf.random_normal(tf.shape(self.mu))
        # self.action = tf.clip_by_value(x, 0.01, 1)

        self.recurrent = False

    def _inverse_action(self, x):
        tmp = (x - 0.01)/0.1
        return tf.math.log(tmp/(1-tmp))

    def _produce_action(self, x):
        return  0.1*tf.nn.sigmoid(x)+0.01

    def create_network(self):
        with tf.variable_scope(self.scope):
            states = tf.placeholder(name='ob',dtype=tf.float32,shape=[None,self.state_dim])

            # out = tf.reshape(tensor = states, shape = [tf.shape(states)[0], self.state_dim, 1])
            # out = tf.layers.conv1d(out,
            #                        filters=10,
            #                        kernel_size = 5)
            # out = tf.layers.max_pooling1d(out,
            #                              pool_size = 4,
            #                              strides = 4)
            # out = tf.layers.flatten(out)
            out = states
            with tf.variable_scope('vf'):
                for i, hidden in enumerate(self.hidden_units):
                    out = tf.layers.dense(out, hidden, name="fc%i" % (i + 1), activation=tf.nn.relu,
                                                          kernel_initializer=self.normc_initializer(1.0))
                value = tf.layers.dense(out, 1, name='final', kernel_initializer=self.normc_initializer(1.0))[:,0]
                self.vpred = value
            with tf.variable_scope('pol'):
                out = states
                for i, hidden in enumerate(self.hidden_units):
                    out = tf.layers.dense(out, hidden, name='fc%i' % (i + 1), activation=tf.nn.relu,
                                                          kernel_initializer=self.normc_initializer(1.0))

                mean = tf.layers.dense(out, self.action_dim, name='final', kernel_initializer=self.normc_initializer(1.0))
                logstd = tf.get_variable(shape=[1,self.action_dim],name = 'std', initializer=tf.zeros_initializer())

        return states,mean,logstd,value


    def get_a_v(self, ob):
        act,value =  self.sess.run((self.action, self.value), feed_dict={
            self.state: [ob]
        })
        return act[0],value[0]


    def get_param(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)

    def get_trainable_param(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)

    def neglogp(self, x):
        # print('x is',x) x is an action
        return 0.5 * tf.reduce_sum(tf.square((self._inverse_action(x) - self.mu) / self.sigma), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1]) \
               + tf.reduce_sum(tf.log(self.sigma), axis=-1)


    def normc_initializer(self, std=1.0, axis=0):
        def _initializer(shape, dtype=None, partition_info=None):
            out = np.random.randn(*shape).astype(dtype.as_numpy_dtype)
            out *= std / np.sqrt(np.square(out).sum(axis=axis, keepdims=True))
            return tf.constant(out)

        return _initializer
