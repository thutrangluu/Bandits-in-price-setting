# -*- coding: utf-8 -*-

# import modules 
import numpy as np 
import pandas as pd
import random
import matplotlib.pyplot as plt 
from eps_bandit import eps_bandit
from ucb_bandit import ucb_bandit
from lin_ucb import eps_bandit_lin, ucb_bandit_lin


## simulate problem
"""
C (int): seats 

T (int): time steps

"""
random.seed(98760)
C = 100
T = 3000
prices = random.sample(range(100, 500), 200)
Arms = list(prices[:C])
Arms_train = list(prices[C:])

def demand_func (price):
    demand=np.zeros(len(price))
    for i,v in enumerate(price):
        a = 120
        b = -0.24
        demand[i] = (np.floor(a + b * v)+np.random.randint(1,5))
    return demand

Sales_prob = demand_func(Arms)/C
Sales_prob_unif= np.random.uniform()


C_rewards = Arms*Sales_prob

lin_bandit_train_data = {
    'arms': Arms_train,
    'mu': demand_func(Arms_train)/C
    }

runs = 1000

bandit_e_05_rw = np.zeros(T)
bandit_e_0_rw = np.zeros(T)
bandit_e_1_rw = np.zeros(T)
bandit_ucb_rw = np.zeros(T)
bandit_e_lin_rw = np.zeros(T)
bandit_ucb_lin_rw = np.zeros(T)

bandit_e_05_action = np.zeros(C)
bandit_e_0_action = np.zeros(C)
bandit_e_1_action = np.zeros(C)
bandit_ucb_action = np.zeros(C)
bandit_e_lin_action = np.zeros(C)
bandit_ucb_lin_action = np.zeros(C)

bandit_e_05_action_total= np.zeros((C,runs))
bandit_e_0_action_total = np.zeros((C,runs))
bandit_e_1_action_total = np.zeros((C,runs))
bandit_ucb_action_total = np.zeros((C,runs))
bandit_e_lin_action_total = np.zeros((C,runs))
bandit_ucb_lin_action_total = np.zeros((C,runs))

bandit_e_05_exploit = 0
bandit_e_0_exploit  = 0
bandit_e_1_exploit  = 0
bandit_e_lin_exploit  = 0

bandit_e_05_explore = 0
bandit_e_0_explore = 0
bandit_e_1_explore = 0
bandit_e_lin_explore = 0

bandit_ucb_bestk  = []
bandit_ucb_lin_bestk = []

# Run experiments
for i in range(runs):
    # Initialize bandits
    bandit_e_05 = eps_bandit(C, 0.05, T, mu=Sales_prob, arms=Arms)
    bandit_e_0 = eps_bandit(C, 0, T, mu=Sales_prob, arms=Arms)
    bandit_e_1 = eps_bandit(C, 0.1, T, mu=Sales_prob, arms=Arms)
    
    bandit_ucb = ucb_bandit(C, T, mu=Sales_prob, arms=Arms, confidence_level=2.0)
    
    bandit_e_lin = eps_bandit_lin(C, 0.05, T,
                                  lin_bandit_train_data, mu=Sales_prob, arms=Arms)
    bandit_ucb_lin = ucb_bandit_lin(C, T,
                                    lin_bandit_train_data, mu=Sales_prob, arms=Arms,confidence_level=0.5)
    
    # Run experiments
    bandit_e_05.run()
    bandit_e_0.run()
    bandit_e_1.run()
    bandit_ucb.run()
    bandit_e_lin.run()
    bandit_ucb_lin.run()

    # Update long-term averages
    bandit_e_05_rw = bandit_e_05_rw + (
        bandit_e_05.reward - bandit_e_05_rw) / (i + 1)
    bandit_e_0_rw = bandit_e_0_rw + (
        bandit_e_0.reward - bandit_e_0_rw) / (i + 1)
    bandit_e_1_rw = bandit_e_1_rw + (
        bandit_e_1.reward - bandit_e_1_rw) / (i + 1)   
    bandit_ucb_rw = bandit_ucb_rw + (
        bandit_ucb.reward - bandit_ucb_rw) / (i + 1)
    bandit_e_lin_rw = bandit_e_lin_rw + (
        bandit_e_lin.reward - bandit_e_lin_rw) / (i + 1)
    bandit_ucb_lin_rw = bandit_ucb_lin_rw + (
        bandit_ucb_lin.reward - bandit_ucb_lin_rw) / (i + 1)
    
    # Average actions per run
    bandit_e_05_action = bandit_e_05_action + (
        bandit_e_05.k_n - bandit_e_05_action) / (i + 1)
    bandit_e_0_action = bandit_e_0_action + (
        bandit_e_0.k_n - bandit_e_0_action) / (i + 1)
    bandit_e_1_action = bandit_e_1_action + (
        bandit_e_1.k_n - bandit_e_1_action) / (i + 1)
    bandit_ucb_action = bandit_ucb_action + (
        bandit_ucb.k_n - bandit_ucb_action) / (i + 1)
    bandit_e_lin_action = bandit_e_lin_action + (
        bandit_e_lin.k_n - bandit_e_lin_action) / (i + 1)
    bandit_ucb_lin_action = bandit_ucb_lin_action + (
        bandit_ucb_lin.k_n - bandit_ucb_lin_action) / (i + 1)
    
    # actions per run
    bandit_e_05_action_total[:,i] = bandit_e_05.k_n
    bandit_e_0_action_total[:,i] = bandit_e_0.k_n
    bandit_e_1_action_total[:,i] = bandit_e_1.k_n
    bandit_ucb_action_total[:,i] = bandit_ucb.k_n
    bandit_e_lin_action_total[:,i] = bandit_e_lin.k_n
    bandit_ucb_lin_action_total[:,i] = bandit_ucb_lin.k_n
    # print(i)

    
    # Average time exploit:
    bandit_e_05_exploit = bandit_e_05_exploit + (
        bandit_e_05.exploit - bandit_e_05_exploit) /(i + 1)
    bandit_e_0_exploit = bandit_e_0_exploit + (
        bandit_e_0.exploit - bandit_e_0_exploit) /(i + 1)
    bandit_e_1_exploit = bandit_e_1_exploit + (
        bandit_e_1.exploit - bandit_e_1_exploit) /(i + 1)
    bandit_e_lin_exploit = bandit_e_lin_exploit + (
        bandit_e_lin.exploit - bandit_e_lin_exploit) /(i + 1)
    
    # Average time explore:
    bandit_e_05_explore = bandit_e_05_explore + (
        bandit_e_05.explore - bandit_e_05_explore) /(i + 1)
    bandit_e_0_explore = bandit_e_0_explore + (
        bandit_e_0.explore - bandit_e_0_explore) /(i + 1)
    bandit_e_1_explore = bandit_e_1_explore + (
        bandit_e_1.explore - bandit_e_1_explore) /(i + 1)
    bandit_e_lin_explore = bandit_e_lin_explore + (
        bandit_e_lin.explore - bandit_e_lin_explore) /(i + 1)
 
    # best price per run
    bandit_ucb_bestk.append(Arms[bandit_ucb.current_k])
    bandit_ucb_lin_bestk.append(Arms[bandit_ucb_lin.current_k])   

## Basic MAB eps and UCB
plt.figure(figsize=(12,8), layout='constrained')
plt.plot(bandit_e_05_rw, label="rewards epsilon = 0.05")
plt.plot(bandit_e_0_rw, label="rewards epsilon = 0")
plt.plot(bandit_e_1_rw, label="rewards epsilon = 0.1")
plt.plot(bandit_ucb_rw, label="rewards ucb")
# plt.plot(bandit_e_lin_rw, label="rewards epsilon lin")
# plt.plot(bandit_ucb_lin_rw, label="rewards ucb lin")
plt.legend(bbox_to_anchor=(1.3, 0.5))
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("Average $\epsilon-greedy$ Rewards vs Average UCB Rewards after " + 
          str(runs) + " runs")
plt.show()

plt.figure(figsize=(12,8), layout='constrained')
plt.plot(bandit_e_05_rw, label="rewards epsilon = 0.05")
plt.plot(bandit_ucb_rw, label="rewards ucb")
plt.plot(bandit_e_lin_rw, label="rewards epsilon lin")
plt.plot(bandit_ucb_lin_rw, label="rewards ucb lin")
plt.legend(bbox_to_anchor=(1.3, 0.5))
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("Average $\epsilon-greedy$ Rewards vs Average UCB Rewards after " + 
          str(runs) + " runs")
plt.show()

action_df=pd.DataFrame(data = np.transpose([Arms,
                                      bandit_e_05_action, 
                                      bandit_e_0_action,
                                      bandit_e_1_action,
                                      bandit_ucb_action, 
                                      bandit_e_lin_action,
                                      bandit_ucb_lin_action,
                                      bandit_e_05_action_total.sum(axis=1),
                                      bandit_e_0_action_total.sum(axis=1),
                                      bandit_e_1_action_total.sum(axis=1),
                                      bandit_ucb_action_total.sum(axis=1),
                                      bandit_e_lin_action_total.sum(axis=1),
                                      bandit_ucb_lin_action_total.sum(axis=1)]), 
                  index = range(C), 
                  columns = ["Price",
                             "Avg actions 0.05",
                             "Avg actions 0",
                             "Avg actions 0.1",
                             "Avg actions UCB",
                             "Avg actions 0.05-lin",
                             "Avg actions UCB-lin",
                             "Total actions 0.05",
                             "Total actions 0",
                             "Total actions 0.1",
                             "Total actions UCB",
                             "Total actions 0.05-lin",
                             "Total actions UCB-lin"])

avg_bestk_ucb = np.mean(bandit_ucb_bestk)
avg_bestk_ucb_lin = np.mean(bandit_ucb_lin_bestk)
avg_bestk_e_05 = action_df[action_df['Avg actions 0.05'] == max(action_df['Avg actions 0.05'])]['Price'].iloc[0]
avg_bestk_e_0 = action_df[action_df['Avg actions 0'] == max(action_df['Avg actions 0'])]['Price'].iloc[0]
avg_bestk_e_1 = action_df[action_df['Avg actions 0.1'] == max(action_df['Avg actions 0.1'])]['Price'].iloc[0]
avg_bestk_e_lin = action_df[action_df['Avg actions 0.05-lin'] == max(action_df['Avg actions 0.05-lin'])]['Price'].iloc[0]

bar_width = 0.5  # Adjust this value to increase/decrease spacing
fig, axs = plt.subplots(4, sharex=True, sharey=True, figsize=(12,8))
fig.suptitle("Actions Selected by Each Algorithm")
axs[0].bar(action_df['Price'], action_df["Avg actions 0.05"],
        color='b', width=bar_width,
        label="$\epsilon=0.5$")
axs[1].bar(action_df['Price'], action_df["Avg actions UCB"],
        color='g',width=bar_width,
        label="UCB")
axs[2].bar(action_df['Price'], action_df["Avg actions 0.05-lin"],
        color='r', width=bar_width,
        label="epsilon lin")
axs[3].bar(action_df['Price'], action_df["Avg actions UCB-lin"],
        color='k', width=bar_width,
        label="ucb lin")
fig.legend(bbox_to_anchor=(1.05, 0.5))
plt.xlabel("Action")
plt.ylabel("Average number of Actions Taken")
plt.show()

opt_per = np.array([bandit_e_05_action, bandit_ucb_action,
                   bandit_e_lin_action,bandit_ucb_lin_action]) / T * 100
df = pd.DataFrame(opt_per, index=['epsilon=0.5', 
    'UCB', 'epsilon lin','UCB lin'],
                 columns=["a = " + str(x) for x in Arms])
print("Percentage of actions selected:")
df
