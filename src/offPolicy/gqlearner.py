import numpy as np

alpha = 0.01
beta = 0.009
gamma = 0.9
epsilon = 0.05
lambd = 0
numberOfFeatures=121
numberOfActions = 4

weights = np.zeros(numberOfFeatures * numberOfActions)
aux_weights = np.zeros(numberOfFeatures * numberOfActions)
e = np.zeros(numberOfFeatures * numberOfActions)


def makeFeaturesVec(features_ids, action):
    result = np.zeros(numberOfFeatures * numberOfActions);
    for id in features_ids:
        result[ action*numberOfFeatures + id] = 1.0
    return result;

def receiveSample(features_current_state, action, reward,features_next_state,proba_action_bpolicy):
    global weights,e,aux_weights,alpha,beta,gamma,epsilon,numberOfFeatures,numberOfActions
    phi_t = makeFeaturesVec(features_current_state,action);
    phi_tnext = makeFeaturesVec(features_next_state, action);
    Q = np.dot(weights,phi_t)
    bestCurrentAction = np.argmax(Q)
    QNext = np.dot(weights, phi_tnext)
    bestNextAction = np.argmax(QNext)

    bar_phi_tnext = np.zeros(numberOfFeatures * numberOfActions);
    for a in range(numberOfActions):
        policy_coeff = epsilon/numberOfActions + (1.0 - epsilon if a==bestNextAction else 0);
        bar_phi_tnext += policy_coeff * makeFeaturesVec(features_next_state, a)


    delta = reward + gamma*np.dot(weights,bar_phi_tnext) - np.dot(weights,phi_t)

    policy_coeff = epsilon/numberOfActions + (1.0 - epsilon if action==bestCurrentAction else 0);
    rho = policy_coeff / proba_action_bpolicy
    
    e = phi_t + rho*gamma*lambd*e

    weights+= alpha*(delta*e - gamma*(1.0 - lambd)*(np.dot(e,weights))*bar_phi_tnext)
    aux_weights+= beta*(delta*e - np.dot(phi_t, weights)*phi_t)


receiveSample(np.array([4]),1,-2,np.array([5]), 0.9)
print(weights)
