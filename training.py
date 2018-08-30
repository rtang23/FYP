import tensorflow as tf             # Deep Learning library
import numpy as np                  # Handle matrices
import random                       # used to see if we explore or exploit
import warnings                     # This ignore all the warning messages that are normally printed during the training because of skiimage
from collections import deque       # Ordered collection with ends
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros         # import Kautenja's gym environment
from gym_super_mario_bros.actions import RIGHT_ONLY
from skimage import transform       # Help us to preprocess the frames
from skimage.color import rgb2gray  # Help us to gray our frames
warnings.filterwarnings('ignore')   # used to ignore warning messages

# Create our environment
env = gym_super_mario_bros.make('SuperMarioBros-v0') # Creates the environment
env = BinarySpaceToDiscreteSpaceEnv(env, RIGHT_ONLY) # have to pick complex movement to try different combos

env.render() # updates the action within the game or pretty much shows you the game is playing

print("The size of our frame is: ", env.observation_space) # was originally a test to see what this was outputting.
print("The action size is : ", env.action_space.n)  # the amount of actions we can take in the game

# Here we create an hot encoded version of our actions
# possible_actions = [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0]...]
possible_actions = np.array(np.identity(env.action_space.n,dtype=int).tolist())


def preprocess_frame(frame):
    # Greyscale frame
    gray = rgb2gray(frame)

    # Crop the screen (remove the part below the player)
    # [Up: Down, Left: right]
    cropped_frame = gray[8:-12, 4:-12]

    # Normalize Pixel Values
    normalized_frame = cropped_frame / 255.0

    # Resize
    preprocessed_frame = transform.resize(normalized_frame, [256, 224])

    return preprocessed_frame  # 110x84x1 frame


stack_size = 2  # We stack 4 frames

# Initialize deque with zero-images one array for each image
stacked_frames = deque([np.zeros((256, 224), dtype=np.int) for i in range(stack_size)], maxlen=2)


def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocess frame
    frame = preprocess_frame(state)

    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((256, 224), dtype=np.int) for i in range(stack_size)], maxlen=2)

        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        #stacked_frames.append(frame)
        #stacked_frames.append(frame)

        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)

    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2)

    return stacked_state, stacked_frames

### MODEL HYPERPARAMETERS
state_size = [256, 224, 2]      # Our input is a stack of 4 frames hence 110x84x4 (Width, height, channels)
action_size = env.action_space.n # 8 possible actions
learning_rate =  0.00025      # Alpha (aka learning rate)

### TRAINING HYPERPARAMETERS
total_episodes = 1000           # Total episodes for training
max_steps = 50000              # Max possible steps in an episode
batch_size = 32                # Batch size

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability
decay_rate = 0.00001           # exponential decay rate for exploration prob


# Q learning hyperparameters
gamma = 0.99                    # Discounting rate

### MEMORY HYPERPARAMETERSde a bet wi
pretrain_length = batch_size   # Number of experiences stored in the Memory when initialized for the first time
memory_size = 10000          # Number of experiences the Memory can keep

### PREPROCESSING HYPERPARAMETERS
stack_size = 2                 # Number of frames stacked

### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = True

## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
episode_render = False


class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate_init = 0.00025
        self.learning_rate_decay_steps = 5
        self.learning_rate_decay = 0.99999
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.learning_rate_init,
                                                        self.global_step,
                                                        self.learning_rate_decay_steps,
                                                        self.learning_rate_decay)
        tf.summary.scalar("learning_rate", self.learning_rate)

        with tf.variable_scope(name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 84, 84, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")

            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            """
            First convnet:
            CNN
            ELU
            """
            # Input is 110x84x4
            self.conv1 = tf.layers.conv2d(inputs=self.inputs_,
                                          filters=8,
                                          kernel_size=[7, 7],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv1")
            self.conv1_relu = tf.nn.relu(self.conv1, name="conv1_relu")
            self.conv1_out = tf.nn.max_pool(self.conv1_relu, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME',name='conv1_out')

            """
            Second convnet:
            CNN
            ELU
            """
            self.conv2 = tf.layers.conv2d(inputs=self.conv1_out,
                                          filters=16,
                                          kernel_size=[5, 5],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv2")
            self.conv2_relu = tf.nn.relu(self.conv2, name="conv2_relu")
            self.conv2_out = tf.nn.max_pool(self.conv2_relu, ksize=[1,3,3,1],strides=[1,2,2,1], padding='SAME',name='conv2_out' )

            """
            Third convnet:
            CNN
            ELU
            """
            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out,
                                          filters=32,
                                          kernel_size=[3, 3],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv3")


            self.conv3_out = tf.nn.relu(self.conv3, name="conv3_out")

            self.flatten = tf.contrib.layers.flatten(self.conv3_out)

            self.fc = tf.layers.dense(inputs=self.flatten,
                                      units=256,
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      name="fc1")

            self.output = tf.layers.dense(inputs=self.fc,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units=self.action_size,
                                          activation=None)

            # Q is our predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_))

            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q), name="loss")
            tf.summary.scalar("loss", self.loss)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)


# Reset the graph
tf.reset_default_graph()

# Instantiate the DQNetwork
DQNetwork = DQNetwork(state_size, action_size, learning_rate)

class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                 size=batch_size,
                                 replace=False)

        return [self.buffer[i] for i in index]


# Instantiate memory
memory = Memory(max_size=memory_size)
for i in range(pretrain_length):
    # If it's the first step
    if i == 0:
        state = env.reset()

        state, stacked_frames = stack_frames(stacked_frames, state, True)

    # Get the next_state, the rewards, done by taking a random action
    choice = random.randint(1, len(possible_actions)) - 1
    action = possible_actions[choice]
    action_ind = np.argmax(action)
    next_state, reward, done, _ = env.step(action_ind)

    # env.render()

    # Stack the frames
    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

    # If the episode is finished (we're dead 3x)
    if done:
        # We finished the episode
        next_state = np.zeros(state.shape)

        # Add experience to memory
        memory.add((state, action, reward, next_state, done))

        # Start a new episode
        state = env.reset()

        # Stack the frames
        state, stacked_frames = stack_frames(stacked_frames, state, True)

    else:
        # Add experience to memory
        memory.add((state, action, reward, next_state, done))

        # Our new state is now the next_state
        state = next_state

# Setup TensorBoard Writer
writer = tf.summary.FileWriter('~/tensorboard/dqn/run')

## Losses
tf.summary.scalar("Loss", DQNetwork.loss)

write_op = tf.summary.merge_all()

"""
This function will do the part
With ϵϵ select a random action atat, otherwise select at=argmaxaQ(st,a)
"""


def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
    ## EPSILON GREEDY STRATEGY
    # Choose action a from state s using epsilon greedy.
    ## First we randomize a number
    exp_exp_tradeoff = np.random.rand()

    # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

    #Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state.reshape((1, *state.shape))})
    # print("Q values are:", Qs)

    if (explore_probability > exp_exp_tradeoff):
        # Make a random action (exploration)
        #print("Random action taken")
        choice = random.randint(1, len(possible_actions)) - 1
        action = possible_actions[choice]

    else:
        # Get action from Q-network (exploitation)
        # Estimate the Qs values state
        #print("Taking the best action")
        Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state.reshape((1, *state.shape))})

        # Take the biggest Q value (= the best action)
        choice = np.argmax(Qs)
        action = possible_actions[choice]

    return action, explore_probability


# Saver will help us to save our model
saver = tf.train.Saver()

with tf.device('/cpu:0'):

    if training == True:
        with tf.Session() as sess:
                # Initialize the variables
            sess.run(tf.global_variables_initializer())

                # Initialize the decay rate (that will use to reduce epsilon)
            decay_step = 0
            for episode in range(total_episodes):
                    # Set step to 0
                step = 0

                    # Initialize the rewards of the episode
                episode_rewards = []

                    # Make a new episode and observe the first state
                state = env.reset()

                    # Remember that stack frame function also call our preprocess function.
                state, stacked_frames = stack_frames(stacked_frames, state, True)

                while step < max_steps:
                    step += 1

                        # Increase decay_step
                    decay_step += 1
                        # Predict the action to take and take it
                    action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state,
                                                                     possible_actions)

                        # Perform the action and get the next_state, reward, and done information
                    action_ind = np.argmax(action)
                    next_state, reward, done, _ = env.step(action_ind)

                    if episode_render:
                        env.render()

                        # Add the reward to total reward
                    episode_rewards.append(reward)

                        # If the game is finished
                    if done:
                            # The episode ends so no next state
                        next_state = np.zeros((256, 224), dtype=np.int)

                        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                            # Set step = max_steps to end the episode
                        step = max_steps

                            # Get the total reward of the episode
                        total_reward = np.sum(episode_rewards)

                        print('Episode: {}'.format(episode),
                                'Total reward: {}'.format(total_reward),
                                'Explore P: {:.4f}'.format(explore_probability),
                                'Training Loss {:.4f}'.format(loss))
                            #print("Episode is:", episode)
                            #print("Total rewards is:", total_reward)
                        rewards_list = []
                        rewards_list.append((episode, total_reward))

                            # Store transition <st,at,rt+1,st+1> in memory D
                        memory.add((state, action, reward, next_state, done))

                    else:
                            # Stack the frame of the next_state
                        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                            # Add experience to memory
                        memory.add((state, action, reward, next_state, done))

                            # st+1 is now our current state
                        state = next_state

                        ### LEARNING PART
                        # Obtain random mini-batch from memory
                    batch = memory.sample(batch_size)
                    states_mb = np.array([each[0] for each in batch], ndmin=3)
                    actions_mb = np.array([each[1] for each in batch])
                    rewards_mb = np.array([each[2] for each in batch])
                    next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                    dones_mb = np.array([each[4] for each in batch])

                    target_Qs_batch = []

                        # Get Q values for next_state
                    Qs_next_state = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: next_states_mb})

                        # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
                    for i in range(0, len(batch)):
                        terminal = dones_mb[i]

                            # If we are in a terminal state, only equals reward
                        if terminal:
                            target_Qs_batch.append(rewards_mb[i])

                        else:
                            target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                            target_Qs_batch.append(target)

                    targets_mb = np.array([each for each in target_Qs_batch])

                    loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer],
                                           feed_dict={DQNetwork.inputs_: states_mb,
                                                      DQNetwork.target_Q: targets_mb,
                                                      DQNetwork.actions_: actions_mb})

                        # Write TF Summaries
                    summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: states_mb,
                                                                DQNetwork.target_Q: targets_mb,
                                                                DQNetwork.actions_: actions_mb})
                    writer.add_summary(summary, episode)
                    writer.flush()

                    # Save model every 5 episodes
                if episode % 5 == 0:
                    save_path = saver.save(sess, "./models/model.ckpt")
                    print("Model Saved")


with tf.Session() as sess:
    total_test_rewards = []
    # Load the model
    saver.restore(sess, "./models/model.ckpt")
    print("Model loaded")

    for episode in range(1):
        total_rewards = 0

        state = env.reset()
        state, stacked_frames = stack_frames(stacked_frames, state, True)

        print("****************************************************")
        print("EPISODE ", episode)

        while True:
            # Reshape the state
            state = state.reshape((1, *state_size))
            # Get action from Q-network
            # Estimate the Qs values state
            Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state})

            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            action = possible_actions[choice]
            action_ind = np.argmax(action)
            # Perform the action and get the next_state, reward, and done information
            next_state, reward, done, _ = env.step(action_ind)
            env.render()

            total_rewards += reward

            if done:
                print ("Score", total_rewards)
                total_test_rewards.append(total_rewards)
                break

            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            state = next_state

    env.close()

