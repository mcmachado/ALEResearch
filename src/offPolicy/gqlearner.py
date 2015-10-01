import numpy as np
import sys
alpha = 0.05
beta = 0
gamma = 0.99
epsilon = 0.05
lambd = 0.5
numberOfFeatures=121
numberOfActions = 4

weights = np.zeros(numberOfFeatures * numberOfActions,dtype=np.float64)
aux_weights = np.zeros(numberOfFeatures * numberOfActions,dtype=np.float64)
e = np.zeros(numberOfFeatures * numberOfActions,dtype=np.float64)


def makeFeaturesVec(features_ids, action):
    result = np.zeros(numberOfFeatures * numberOfActions);
    for id in features_ids:
        result[ action*numberOfFeatures + id] = 1.0
    return result;

def receiveSample(features_current_state, action, reward,features_next_state,proba_action_bpolicy):
    global weights,e,aux_weights,alpha,beta,gamma,epsilon,numberOfFeatures,numberOfActions
    #print("curF %d nextF %d" % (features_current_state[0], features_next_state[0]))
    phi_t = makeFeaturesVec(features_current_state,action);
    phi_tnext = makeFeaturesVec(features_next_state, action);
    Q = np.zeros(numberOfActions)
    QNext = np.zeros(numberOfActions)
    for i in range(numberOfActions):
        for j in range(numberOfFeatures):
            Q[i]+=weights[i*numberOfFeatures + j]*(1.0 if j in features_current_state else 0)
            QNext[i]+=weights[i*numberOfFeatures + j]*(1.0 if j in features_next_state else 0)
    #print("currQ : ",end="")
    #for q in Q:
        #print(q,end=" ")
    #print("")
    #print("nextQ : ",end="")
    #for q in QNext:
        #print(q,end=" ")
    #print("")
    bestCurrentAction = np.argmax(Q)
    bestNextAction = np.argmax(QNext)
    #print("currA %d" % (bestCurrentAction))
    #print("nextA %d" % (bestNextAction))
    bar_phi_tnext = np.zeros(numberOfFeatures * numberOfActions);
    for a in range(numberOfActions):
        policy_coeff = epsilon/numberOfActions + (1.0 - epsilon if a==bestNextAction else 0);
        bar_phi_tnext += policy_coeff * makeFeaturesVec(features_next_state, a)

    #print("theta * bar_phi %f" % (np.dot(weights,bar_phi_tnext)))
    #print("theta * phi %f" % (np.dot(weights,phi_t)))
    
    delta = reward + gamma*np.dot(weights,bar_phi_tnext) - np.dot(weights,phi_t)

    #print("delta %f" % (delta))
    policy_coeff = epsilon/numberOfActions + (1.0 - epsilon if action==bestCurrentAction else 0);
    rho = policy_coeff / proba_action_bpolicy
    #print("rho %f" % (rho))
    
    e = phi_t + rho*gamma*lambd*e
    #print("e :")
    #for i in range(numberOfActions):
    #    for j in range(numberOfFeatures):
    #        if e[i*numberOfFeatures + j]!=0:
                #print("action %d id %d value %f" % (i,j,e[i*numberOfFeatures +j]))

    #print("e*weights %e" % (np.dot(e,aux_weights)))

    weights+= alpha*(delta*e - gamma*(1.0 - lambd)*(np.dot(e,aux_weights))*bar_phi_tnext)
    #print("phi*w %f" % (np.dot(phi_t, aux_weights)))
    aux_weights+= beta*(delta*e - np.dot(phi_t, aux_weights)*phi_t)
    #print("aux_weights")
    #for i in range(numberOfActions):
    #    for j in range(numberOfFeatures):
    #        if aux_weights[i*numberOfFeatures + j]==0:
                #print("0 ", end='')
    #        else:
                #print("%f " % (aux_weights[i*numberOfFeatures + j]), sep='' ,end='')
        #print("")
    #print("weights")
    #for i in range(numberOfActions):
    #    for j in range(numberOfFeatures):
    #        if weights[i*numberOfFeatures + j]==0:
                #print("0 ", end='')
    #        else:
                #print("%g " % (weights[i*numberOfFeatures + j]), sep='' ,end='')
        #print("")



#receiveSample(np.array([4]),1,-2,np.array([5]), 0.9)
#receiveSample(np.array([10]),2,-2,np.array([5]), 0.9)
#receiveSample(np.array([2]),3,-2,np.array([5]), 0.9)
#receiveSample(np.array([30]),0,-2,np.array([5]), 0.9)
#receiveSample(np.array([40]),1,-2,np.array([5]), 0.9)
#receiveSample(np.array([120]),2,-2,np.array([5]), 0.9)

file = open("samples.txt","r")
it = 0
for line in file:
    data = line.split()
    if it>30000:
        break
    receiveSample(np.array([int(data[0])]), int(data[1]), float(data[2]), np.array([int(data[3])]), float(data[4]))
    it += 1


print("")
print("")
for i in range(11):
    for j in range(11):
        idx = j + i*11;
        act = 0;
        for a in range(numberOfActions):
            if(weights[a*numberOfFeatures+idx] > weights[act*numberOfFeatures+idx]):
                act = a;
                
        if act==0:
            print("^",end='\t')
        elif act==1:
            print("|",end='\t')
        elif act==2:
            print("<",end='\t')
        elif act==3:
            print(">",end='\t')
    print("")

for i in range(numberOfActions):
    for j in range(numberOfFeatures):
        if weights[i*numberOfFeatures + j]==0:
            print("0 ", end='', file=sys.stderr)
        else:
            print("%f " % (weights[i*numberOfFeatures + j]), sep='' ,end='', file=sys.stderr)
    print("")
    
