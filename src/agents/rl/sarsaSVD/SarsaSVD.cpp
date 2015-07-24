/****************************************************************************************
 ** Implementation of Sarsa(lambda). It implements Fig. 8.8 (Linear, gradient-descent
 ** Sarsa(lambda)) from the book "R. Sutton and A. Barto; Reinforcement Learning: An
 ** Introduction. 1st edition. 1988."
 ** Some updates are made to make it more efficient, as not iterating over all features.
 **
 ** TODO: Make it as efficient as possible.
 **
 ** Author: Marlos C. Machado
 ***************************************************************************************/

#include "../../../common/Timer.hpp"
#include "SarsaSVD.hpp"
#include <stdio.h>
#include <math.h>
#include <random>

using namespace Eigen;

SarsaSVD::SarsaSVD(Environment<bool>& env, Parameters *param, unsigned nFlavors) : RLLearner<bool>(env, param) {
    delta = 0.0;

    alpha = param->getAlpha();
    lambda = param->getLambda();
    traceThreshold = param->getTraceThreshold();
    numFeatures = env.getNumberOfFeatures();
    toSaveWeightsAfterLearning = param->getToSaveWeightsAfterLearning();
    saveWeightsEveryXSteps = param->getFrequencySavingWeights();
    pathWeightsFileToLoad = param->getPathToWeightsFiles();

    numFlavors = nFlavors;

    w = MatrixXf::Zero(numFlavors,numFeatures*numActions);
    // std::mt19937 g1(param->getSeed());
    // std::normal_distribution<float> distribution(0.0,0.1);
    // for(unsigned i = 0;i<w.rows();i++){
    //     for(unsigned j = 0; j<w.cols();j++){
    //         w(i,j) = distribution(g1);
    //     }
    // }
    
    e = VectorXf::Zero(numFeatures*numActions);

    for(int i = 0; i < numActions; i++){
        //Initialize Q;
        Q.push_back(0);
        Qnext.push_back(0);
    }

    if(toSaveWeightsAfterLearning){
        std::stringstream ss;
        ss << param->getFileWithWeights() << param->getSeed() << ".wgt";
        nameWeightsFile =  ss.str();
    }

    if(param->getToLoadWeights()){
        loadWeights();
    }
}

SarsaSVD::~SarsaSVD(){}

void SarsaSVD::updateQValues(std::vector<int> &Features, std::vector<float> &QValues, unsigned curFlavor){
    for(int a = 0; a < numActions; a++){
        float sumW = 0;
        for(unsigned int i = 0; i < Features.size(); i++){
            sumW += w(curFlavor,a*numFeatures+Features[i]);
        }
        QValues[a] = sumW;
    }
}

void SarsaSVD::updateReplTrace(int action, std::vector<int> &Features){
    //e <- gamma * lambda * e
    e *= gamma*lambda;
    //For all i in Fa:
    for(unsigned int i = 0; i < F.size(); i++){
        int idx = Features[i];
        e[action*numFeatures + idx] = 1;
    }
}

void SarsaSVD::sanityCheck(){
    for(int i = 0; i < numActions; i++){
        if(fabs(Q[i]) > 10e7 || Q[i] != Q[i] /*NaN*/){
            printf("It seems your algorithm diverged!\n");
            exit(0);
        }
    }
}

void SarsaSVD::saveWeightsToFile(std::string suffix){
    std::ofstream weightsFile ((nameWeightsFile + suffix).c_str());
    if(weightsFile.is_open()){
        weightsFile << numFlavors << " " << numActions*numFeatures << std::endl;
        for(unsigned int i = 0; i < numFlavors; i++){
            for(unsigned int j = 0; j < numActions*numFeatures; j++){
                if(w(i,j) != 0){
                    weightsFile << i << " " << j << " " << w(i,j) << std::endl;
                }
            }
        }
        weightsFile.close();
    }
    else{
        printf("Unable to open file to write weights.\n");
    }
}

void SarsaSVD::loadWeights(std::string fname){
    std::string line;
    unsigned dim1,dim2;
    int i, j;
    float value;

    std::ifstream weightsFile (fname.c_str());

    weightsFile >> dim1 >> dim2;
    assert(dim1 == numFlavors);
    assert(dim2 == numFeatures*numActions);

    while(weightsFile >> i >> j >> value){
        w(i,j) = value;
    }
}
void SarsaSVD::loadWeights(){
    loadWeights(pathWeightsFileToLoad.c_str());
}

void SarsaSVD::learnPolicy(Environment<bool>& env){

    struct timeval tvBegin, tvEnd, tvDiff;
    std::vector<double> reward;
    double elapsedTime;
    float cumReward = 0, prevCumReward = 0;
    unsigned int maxFeatVectorNorm = 1;
    sawFirstReward = 0; firstReward = 1.0;

    //Repeat (for each episode):
    int episode, totalNumberFrames = 0;

    //whether we saw a positive reward so far
    bool started_learning=false;
    const unsigned deltaCapa = 50;
    std::vector<std::array<float,deltaCapa>> deltas(numFlavors);
    unsigned curDeltaPos=0; //current position in the deltas array
    unsigned numDeltas = 0; //number of deltas stored for the current rank.
    std::vector<float> deltaSum(numFlavors,0.0);
    std::vector<float> lastReward(numFlavors,0.0);

    const unsigned meanCapa = 50;
    std::vector<std::array<float,meanCapa>> means(numFlavors);
    unsigned curMeanPos=0; //current position in the deltas array
    unsigned numMeans = 0; //number of deltas stored for the current rank.
    std::vector<float> meanSum(numFlavors,0.0);
    
    //initial rank
    unsigned rank = 1;
    U = MatrixXf::Ones(numFlavors,rank);
    V = MatrixXf::Ones(numFeatures*numActions,rank);

    std::mt19937 g1(3);
    std::normal_distribution<float> distribution(0.0,0.1);
    // for(unsigned i = 0;i<U.rows();i++){
    //     for(unsigned j = 0; j<U.cols();j++){
    //         U(i,j) = distribution(g1);
    //     }
    // }
    for(unsigned i = 0;i<V.rows();i++){
        for(unsigned j = 0; j<V.cols();j++){
            V(i,j) = distribution(g1);
        }
    }

    
    S = MatrixXf::Zero(rank,rank);
    //S(0,0) = 1;
    //w = U*S*V.transpose();
    
    bool rankIncreaseNeeded = false;
    for(episode = 0; totalNumberFrames < totalNumberOfFramesToLearn; episode++){
        //Pick the current flavor. Be carefull, the rest of the code assumes that they are picked in increasing order, one by one.
        int currentFlavor = episode % numFlavors;
        env.setFlavor(currentFlavor);

        //clear e
        e.setZero(numActions*numFeatures);

        F.clear();

        env.getActiveFeaturesIndices(F);
        updateQValues(F, Q, currentFlavor);
        currentAction = epsilonGreedy(Q);
        //Repeat(for each step of episode) until game is over:
        gettimeofday(&tvBegin, NULL);

        //This also stops when the maximum number of steps per episode is reached
        while(!env.isTerminal()){
            reward.clear();
            reward.push_back(0.0);
            reward.push_back(0.0);
            updateQValues(F, Q,currentFlavor);

            sanityCheck();
            //Take action, observe reward and next state:
            act(env, currentAction, reward);
            cumReward  += reward[1];
            if(!env.isTerminal()){
                //Obtain active features in the new state:
                Fnext.clear();
                env.getActiveFeaturesIndices(Fnext);
                updateQValues(Fnext, Qnext, currentFlavor);     //Update Q-values for the new active features
                nextAction = epsilonGreedy(Qnext);
            }
            else{
                nextAction = 0;
                for(unsigned int i = 0; i < Qnext.size(); i++){
                    Qnext[i] = 0;
                }
            }
            //To ensure the learning rate will never increase along
            //the time, Marc used such approach in his JAIR paper
            if (F.size() > maxFeatVectorNorm){
                maxFeatVectorNorm = F.size();
            }

            delta = reward[0] + gamma * Qnext[nextAction] - Q[currentAction];

            updateReplTrace(currentAction, F);
            //Update weights vector:
            //vector of flavor selection
            VectorXf m = VectorXf::Zero(numFlavors);
            m[currentFlavor] = (alpha/maxFeatVectorNorm) * delta;

            //rank 1 update
            MatrixXf K = MatrixXf::Zero(rank+1,rank+1);
            K.block(0,0,rank,rank) = S;

            VectorXf A = VectorXf::Zero(rank+1);
            VectorXf gram1 = (U.transpose()) * m;
            A.head(rank) = gram1;
            VectorXf p = (m - U*gram1);
            float Ra = p.norm();
            A(rank) = Ra;

            VectorXf B = VectorXf::Zero(rank+1);
            VectorXf gram2 = (V.transpose()) * e;
            B.head(rank) = gram2;
            VectorXf q = (e - V*gram2);
            float Rb = q.norm();
            B(rank) = Rb;

            K += A*B.transpose();

            //compute SVD
            JacobiSVD<MatrixXf> svd(K, ComputeFullU | ComputeFullV);

            //update left space
            MatrixXf augU = MatrixXf::Zero(numFlavors,rank+1);
            augU.topLeftCorner(numFlavors,rank) = U;
            if(Ra!=0)
                augU.bottomRightCorner(numFlavors,1) = p/Ra;

            //update right space
            MatrixXf augV = MatrixXf::Zero(numActions*numFeatures,rank+1);
            augV.topLeftCorner(numActions*numFeatures,rank) = V;
            if(Rb!=0)
                augV.bottomRightCorner(numActions*numFeatures,1) = q/Rb;

            if(rankIncreaseNeeded){
                rankIncreaseNeeded = false;
                //update singular values (keep all rank+1 values)
                S = svd.singularValues().asDiagonal();
                U = augU * svd.matrixU();
                V = augV * svd.matrixV();
                rank++;
            }else{
                //update singular values (keep only |rank| values)
                S = svd.singularValues().head(rank).asDiagonal();
                U = (augU * svd.matrixU()).topLeftCorner(numFlavors,rank);
                V = (augV * svd.matrixV()).topLeftCorner(numActions*numFeatures,rank);
            }

            w = U*S*V.transpose();

            F = Fnext;
            currentAction = nextAction;
        }
        gettimeofday(&tvEnd, NULL);
        timeval_subtract(&tvDiff, &tvEnd, &tvBegin);
        elapsedTime = double(tvDiff.tv_sec) + double(tvDiff.tv_usec)/1000000.0;

        double fps = double(env.getEpisodeFrameNumber())/elapsedTime;
        printf("episode: %d,\t%.0f points,\tavg. return: %.1f,\t%d frames,\t%.0f fps,\t flavor %d\n",
               episode + 1, cumReward - prevCumReward, (double)cumReward/(episode + 1.0),
               env.getEpisodeFrameNumber(), fps,currentFlavor);

        started_learning |= (cumReward - prevCumReward) > 0;
        if(started_learning){
            if(numMeans < meanCapa){
                meanSum[currentFlavor] += (cumReward - prevCumReward);
                means[currentFlavor][curMeanPos] += (cumReward - prevCumReward);
                if(currentFlavor == numFlavors - 1){
                    curMeanPos = (curMeanPos + 1) % meanCapa;
                    numMeans++;
                }                
            }else{
                meanSum[currentFlavor] -= means[currentFlavor][curMeanPos];
                means[currentFlavor][curMeanPos] = (cumReward - prevCumReward);
                meanSum[currentFlavor] += means[currentFlavor][curMeanPos];
                if(currentFlavor == numFlavors - 1){
                    curMeanPos = (curMeanPos + 1) % meanCapa;
                }                
                
                float curDelta = meanSum[currentFlavor]/double(meanCapa) - lastReward[currentFlavor];
                lastReward[currentFlavor] = meanSum[currentFlavor]/double(meanCapa);
                if(numDeltas == deltaCapa){
                    deltaSum[currentFlavor] -= deltas[currentFlavor][curDeltaPos];
                    deltas[currentFlavor][curDeltaPos] = curDelta;
                    deltaSum[currentFlavor] += deltas[currentFlavor][curDeltaPos];
                    if(currentFlavor == numFlavors - 1){
                        curDeltaPos = (curDeltaPos+1) % deltaCapa;
                    }
                }else{
                    deltas[currentFlavor][curDeltaPos] = curDelta;
                    deltaSum[currentFlavor] += curDelta;
                    if(currentFlavor == numFlavors - 1){
                        curDeltaPos = (curDeltaPos + 1) % deltaCapa;
                        numDeltas++;
                    }
                }

                if(numDeltas == deltaCapa && currentFlavor==numFlavors-1 && rank<numFlavors){
                    //if we have gone through all the flavors, we can compute the mean derivative
                    float derivativeSum = 0.0;
                    for(unsigned i = 0;i<numFlavors;i++){
                        std::cout<<deltaSum[i]<<std::endl;
                        derivativeSum += deltaSum[i]/double(deltaCapa);
                    }
                    float meanDerivative = derivativeSum / double(numFlavors);
                    std::cout<<"Derivative "<<meanDerivative<<std::endl;
                    if(meanDerivative < 0.05){
                        std::fill(deltaSum.begin(),deltaSum.end(),0.0);
                        numDeltas = 0;
                        rankIncreaseNeeded = true;
                        std::cout<<"Switching to rank "<<rank+1<<" on episode "<<episode<<std::endl;
                    }
                
                }
            }
        }
        
        totalNumberFrames += env.getEpisodeFrameNumber();
        prevCumReward = cumReward;
        env.reset();
        if(toSaveWeightsAfterLearning && episode%saveWeightsEveryXSteps == 0 && episode > 0){
            std::stringstream ss;
            ss << episode;
            saveWeightsToFile(ss.str());
        }
    }
    if(toSaveWeightsAfterLearning){
        std::stringstream ss;
        ss << episode;
        saveWeightsToFile(ss.str());
    }
}

double SarsaSVD::evaluatePolicy(Environment<bool>& env){
    return evaluatePolicy(env,numEpisodesEval);
}
double SarsaSVD::evaluatePolicy(Environment<bool>& env,unsigned numSteps, bool epsilonAnneal){
    float totR = 0.0;
    //Repeat (for each episode):
    for(int curFlavor = 0; curFlavor<numFlavors; curFlavor++){
        env.setFlavor(curFlavor);
        float reward = 0;
        float cumReward = 0;
        float prevCumReward = 0;
        if(epsilonAnneal)
            epsilon = 1.0;
        for(int episode = 0; episode < numSteps; episode++){
            //Repeat(for each step of episode) until game is over:
            for(int step = 0; !env.isTerminal() && step < episodeLength; step++){
                //Get state and features active on that state:
                F.clear();
                env.getActiveFeaturesIndices(F);
                updateQValues(F, Q, curFlavor);       //Update Q-values for each possible action
                currentAction = epsilonGreedy(Q);
                //compute proba of taking current action
                //first, we need the number of QValues that are tied
                double numTies = 0;
                if(!randomActionTaken){
                    for(const auto& q : Q){
                        if(q==Q[currentAction])
                            numTies++;
                    }
                }

                double proba_action = epsilon/double(numActions) + (randomActionTaken ? 0 : (1.0 - epsilon)/numTies);
                //Take action, observe reward and next state:
                reward = env.act(actions[currentAction],proba_action);
                cumReward  += reward;
            }
            double fps = 0;

            printf("flavor: %d,\t episode: %d,\t%.0f points,\tavg. return: %.1f,\t%d frames,\t%.0f fps epsilon : %f\n", curFlavor,
                   episode + 1, (cumReward-prevCumReward), (double)cumReward/(episode + 1.0), env.getEpisodeFrameNumber(), fps,epsilon);

            env.reset();
            prevCumReward = cumReward;
            if(epsilonAnneal)
                epsilon-=1/(double)(numSteps);
        }
        std::cerr<<"flavor "<<curFlavor<<"\t"<<cumReward/(double)(numSteps)<<std::endl;
        totR += cumReward/(double)(numSteps);
    }
    return totR/(double)(numFlavors);
}
