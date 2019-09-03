import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
GAP_SIZE = 100
er = np.load('rewards/Episode_rewards0.971-time-1567535335.4813647.npy')
print(er.shape)
# plt.scatter(er[:,0])
# plt.show()
rewards = []
epsilon = []
accuracy = []

for i in range(0 , er.shape[0]-GAP_SIZE , GAP_SIZE):
    sum = 0
    sum_1 = 0
    sum_2 = 0
    for j in range(0,GAP_SIZE):
        sum += er[i+j,0]
        sum_1 += er[i+j,1]
        sum_2 += er[i+j,2]
    rewards.append(sum)
    epsilon.append(sum_1)
    accuracy.append(sum_2)
# plt.scatter(x = range(0,er.shape[0]-GAP_SIZE , GAP_SIZE) , y=rewards ,s = 4 , c='lime', marker = 'o' )
# plt.show()
# plt.subplot(2,2,1)
plt.title('Rewards')
plt.plot(rewards , c = 'lime' , marker = '.' , ls = '-')
plt.savefig('rewards/Rewards.png')
plt.show()
# plt.subplot(2,2,2)
plt.title('Epsilon')
plt.plot(epsilon , c = 'red' , marker = '.' , ls = '-')
plt.savefig('rewards/Epsilon.png')
plt.show()
# plt.subplot(2,2,3)
plt.title('Accuracy')
plt.plot(accuracy , c = 'blue' , marker = '.' , ls = '-')
plt.savefig('rewards/Accuracy.png')
plt.show()
# plt.plot(er[:,1] ,  c = 'lime' , marker = '.')
# plt.show()
# plt.plot(er[:,2] ,  c = 'lime'  , marker = '.',ls = '-' )
# plt.show()