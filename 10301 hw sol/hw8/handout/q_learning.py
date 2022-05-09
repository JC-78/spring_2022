import argparse
import numpy as np
from environment import MountainCar, GridWorld
import random
import matplotlib.pyplot as plt

# NOTE: We highly recommend you to write functions for...
# - converting the state to a numpy array given a sparse representation
# - determining which state to visit next


def main(args):
    # Command line inputs
    mode = args.mode
    weight_out = args.weight_out
    returns_out = args.returns_out
    episodes = args.episodes
    max_iterations = args.max_iterations
    epsilon = args.epsilon
    gamma = args.gamma
    learning_rate = args.learning_rate
    debug = args.debug

    # We will initialize the environment for you:
    if args.environment == 'mc':
        env = MountainCar(mode=mode, debug=debug)
    else:
        env = GridWorld(mode=mode, debug=debug)

    weights=np.zeros((env.state_space,env.action_space)) # Our shape is |A| x |S|, if this helps
    bias=0 # If you decide to fold in the bias (hint: don't), recall how the bias is defined!
    returns = np.zeros((episodes))  # This is where you will save the return after each episode
    roll_returns = np.zeros((episodes-25))
    count=0
    count1=0
    for episode in range(episodes):
        if episode>=25:
            #print("count1 is",count1)
            roll_returns[count1]=np.mean(returns[0+count1:25+count1])
            count1+=1
        # Reset the environment at the start of each episode
        state = env.reset()  # `state` now is the initial state
        k=0
        for it in range(max_iterations):
            
            rand=random.random()
            prev=np.zeros((env.state_space,1))
            for item in state:
                prev[item]=state[item]
                
            if rand<=epsilon:
                action=random.randint(0,env.action_space-1) #explore
                a=np.transpose(prev)
                b=weights[:,action]
                max_val=np.matmul(a,b)+bias
            else:
                action=0
                res=[]
                for move in range(3):
                    a=np.transpose(prev)
                    b=weights[:,move]
                    c=np.matmul(a,b)+bias
                    res.append(c)
                action=np.argmax(res)
                max_val=res[action]

            state1,reward,done=env.step(action)
            k+=reward
            action1=0
            max_val1= -1

            state1_1=[]
            for item in state1:
                state1_1.append(state1[item])
            res1=[]

            state1_1=np.zeros((env.state_space,1))
            for item in state1:
                state1_1[item]=state1[item]

            for move in range(3):
                a=np.transpose(state1_1)
                b=weights[:,move]
                c=np.matmul(a,b)+bias
                res1.append(c)
            action1=np.argmax(res1)
            max_val1=res1[action1]
            
            state=state1
            #updating weights and bias
            item=(max_val-(reward+gamma*max_val1))
            grad=np.zeros((env.state_space,env.action_space))
            
            grad[:,action]=prev.flatten()
            weights=weights-learning_rate*item*grad
            bias=bias-learning_rate*item*1

            if done:
                break
        #print("k is ",k)
        returns[count]=k
        count+=1
    x = np.linspace(0, episodes - 1,episodes)
    #x=[i for i in range(1,episodes+1)]
    #print("x is ",x)
    print("returns is ",returns)
    #print("x shape is ",x.shape)
    print("return shape is ", returns.shape)
    plt.xlabel("# of episode")
    plt.ylabel("Returns")
    plt.plot(x[:episodes], returns[:episodes], "r",label = "Returns")
    #plt.plot(x, np.squeeze(returns), "r",label = "Returns")

    #print(len(roll_returns))
    #print("roll returns is ",roll_returns)
    plt.plot(x[25:episodes],roll_returns[:],"b",linewidth=3.0,alpha=0.5,label="Rolling mean")
    plt.title("The relationship between episodes and return")
    plt.legend(loc='upper right')
    plt.show()
    with open(weight_out, 'w') as out:
        out.write(str(bias[0])+"\n")
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                item=str(weights[i][j])+"\n"
                out.write(item)

    with open(returns_out, 'w') as out1:
        for thing in returns:
            item=str(thing)+"\n"
            out1.write(item)
            
if __name__ == "__main__":
    # No need to change anything here
    parser = argparse.ArgumentParser()
    parser.add_argument('environment', type=str, choices=['mc', 'gw'],
                        help='the environment to use')
    parser.add_argument('mode', type=str, choices=['raw', 'tile'],
                        help='mode to run the environment in')
    parser.add_argument('weight_out', type=str,
                        help='path to output the weights of the linear model')
    parser.add_argument('returns_out', type=str,
                        help='path to output the returns of the agent')
    parser.add_argument('episodes', type=int,
                        help='the number of episodes to train the agent for')
    parser.add_argument('max_iterations', type=int,
                        help='the maximum of the length of an episode')
    parser.add_argument('epsilon', type=float,
                        help='the value of epsilon for epsilon-greedy')
    parser.add_argument('gamma', type=float,
                        help='the discount factor gamma')
    parser.add_argument('learning_rate', type=float,
                        help='the learning rate alpha')
    parser.add_argument('--debug', type=bool, default=False,
                        help='set to True to show logging')
    main(parser.parse_args())
