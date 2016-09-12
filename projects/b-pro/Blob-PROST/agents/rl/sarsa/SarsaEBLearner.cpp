/*******************************************************************************
 ** Implementation of Sarsa(lambda). It implements Fig. 8.8 (Linear,
 ** gradient-descent Sarsa(lambda)) from the book "R. Sutton and A. Barto;
 ** Reinforcement Learning: An Introduction. 1st edition. 1988." Some updates
 ** are made to make it more efficient, as not iterating over all features.
 **
 ** TODO: Make it as efficient as possible.
 **
 ** Author: Marlos C. Machado
 ******************************************************************************/

#ifndef TIMER_H
#define TIMER_H
#include "../../../common/Timer.hpp"
#include "../../../common/Mathematics.hpp"
#endif
#include "SarsaEBLearner.hpp"
#include <stdio.h>
#include <math.h>
#include <set>

#include <algorithm>

using namespace std;

SarsaEBLearner::SarsaEBLearner(ALEInterface& ale,
                               Features* features,
                               Parameters* param,
                               int seed)
    : SarsaLearner(ale, features, param, seed) {
  printf("SarsaEBLearner is Running the show!!!\n");
  beta = param->getBeta();
  sigma = param->getSigma();
  kappa = param->getKappa();
  is_min_prob_activated = false;

  init_w_value = beta / (sqrt(kappa) * (1 - gamma));

  for (int i = 0; i < numActions; i++) {
    // Initialize Q;
    QI.push_back(0);
    QInext.push_back(0);
    // Initialize w:
    QI_w.push_back(vector<float>());
  }

  actionMarginals.clear();
  featureProbs.clear();
  featureProbs.reserve(60000);

  NUM_PHI_OFFSET = ACTION_OFFSET + numActions;
}

void SarsaEBLearner::updateQIValues(vector<long long>& Features,
                                    vector<float>& QValues) {
  unsigned long long featureSize = Features.size();
  for (int a = 0; a < numActions; ++a) {
    float sumW = 0;
    for (unsigned long long i = 0; i < featureSize; ++i) {
      sumW = sumW + QI_w[a][Features[i]] * groups[Features[i]].numFeatures;
    }
    QValues[a] = sumW;
  }
}

void SarsaEBLearner::update_action_marginals(int cur_action, int time_step) {
  for (int action = 0; action < numActions; action++) {
    actionMarginals[action] *= (time_step + 1.0) / (time_step + 2);
    if (action == cur_action) {
      actionMarginals[action] += 1.0 / (time_step + 2);
    }
  }
}

void SarsaEBLearner::update_phi(vector<long long>& features, long time_step) {
  // Updating the p(phi) and p(a/phi)
  // p(phi)  : rho_{t+1} = ((rho_t * (t + 1)) + phi_{t + 1}) / (t + 2)

  // Partial Update for all the seen features.
  // rho_{t+1}^' = rho_t * (t + 1) / (t + 2)
  for (auto it = featureProbs.begin(); it != featureProbs.end(); ++it) {
    it->second[0] *= ((time_step + 1.0) / (time_step + 2));
  }

  // Partial Update only for all the currently active features.
  // rho_{t + 1} = rho_{t + 1}^'  + (phi_{t + 1} / (t + 2))
  for (long long featIdx : features) {
    featureProbs[featIdx][0] += (1.0 / (time_step + 2));
  }
}

void SarsaEBLearner::update_action_given_phi(
    unordered_map<long long, vector<double>>& tmp_featureProbs,
    vector<long long>& features,
    int action,
    long time_step) {
  // p(a/phi): p_{t + 1}(a / phi) =
  //    p_t * (n_{phi} + 1) / (n_{phi} + 2) + I[a = cur_act] / (n_{phi} + 2)
  double n_phi = 0;
  for (long long featIdx : features) {
    n_phi = tmp_featureProbs[featIdx][NUM_PHI_OFFSET];
    for (int a = 0; a < numActions; a++) {
      tmp_featureProbs[featIdx][a + ACTION_OFFSET] *=
          (n_phi + 1.0) / (n_phi + 2);
      if (a == action) {
        tmp_featureProbs[featIdx][a + ACTION_OFFSET] += (1.0 / (n_phi + 2));
      }
    }
  }
}

void SarsaEBLearner::add_new_feature_to_map(long long featIdx, int time_step) {
  // Creating new vector to store needed data for active feature.
  vector<double> v(ACTION_OFFSET + numActions + 1);
  v[0] = 1.001 / (time_step + 1);
  v[1] = 0;

  // p(a=cur_act/phi_i=1) = \frac{n_{cur_act} + (1/numActions)}{n_phi + 1}
  // Here, n_{a} = 0
  for (int action = 0; action < numActions; action++) {
    v[action + ACTION_OFFSET] = 1.0 / numActions;
  }
  v[NUM_PHI_OFFSET] = 0;
  featureProbs.insert(std::make_pair(featIdx, v));
}

double SarsaEBLearner::get_sum_log_phi(vector<long long>& features,
                                       long time_step,
                                       bool isFirst) {
  double sum_log_phi = 0;

  // Iterating over the features to calculate the joint
  // TODO: Don't store the last p(a/phi_i),
  //       calculate it as 1 - \sum_{j=1}^{numActions-1} p(a_j/phi_i)
  for (long long featIdx : features) {
    if (featureProbs.find(featIdx) == featureProbs.end()) {
      add_new_feature_to_map(featIdx, time_step);
    }

    // p(phi_i=1)
    sum_log_phi += log(featureProbs[featIdx][0]);

    if (isFirst) {
      // Increment n_{phi} as the feature is active.
      featureProbs[featIdx][NUM_PHI_OFFSET] += 1;
    }

    // Set the feature as seen.
    featureProbs[featIdx][1] = 1;
  }

  // p(phi_i=0)
  for (auto it = featureProbs.begin(); it != featureProbs.end(); ++it) {
    if (it->second[1] == 0) {
      sum_log_phi += log(1 - it->second[0]);
    } else {
      // Reset the seen features to unseen as its update has already been
      // done.
      it->second[1] = 0;
    }
  }

  return sum_log_phi;
}

double SarsaEBLearner::get_sum_log_action_given_phi(
    unordered_map<long long, vector<double>>& context_featureProbs,
    vector<long long>& features,
    int action,
    long time_step) {
  double sum_log_action_given_phi = 0;

  // ASSUMPTION: New features has been added to featureProbs in function
  //  get_sum_log_phi

  for (long long featIdx : features) {
    // p(a=cur_act/phi_i=1)
    sum_log_action_given_phi +=
        log(context_featureProbs[featIdx][action + ACTION_OFFSET]);
    // Set the feature as seen.
    context_featureProbs[featIdx][1] = 1;
  }
  double tmp;
  // p(a=cur_act/phi_i=0) =
  //        (p(a=cur_act) - p(a=cur_act/phi_i=1)*p(phi_i=1))/(1-phi_i=1)
  for (auto it = context_featureProbs.begin(); it != context_featureProbs.end();
       ++it) {
    // Update the probabilities for the inactive features.
    if (it->second[1] == 0) {
      // TODO: fix underfow issue, and address nan! Still might happen.
      tmp = log(actionMarginals[action] -
                (it->second[0] * it->second[action + ACTION_OFFSET])) -
            log(1 - it->second[0]);
      if (tmp != tmp) {
        // printf("(1.2) Crazy difference: %.10f\n",
        //        actionMarginals[action] -
        //            (it->second[0] * it->second[action + ACTION_OFFSET]));
        // printf("(1.3) log(1-p(phi=1)): %f\n", log(1 - it->second[0]));
        // printf("(1.4) p(a)[%d]: %f\n", action, actionMarginals[action]);
        // printf("(1.5) p(a,phi=1)[%d]: %f\n", action,
        //        (it->second[0] * it->second[action + ACTION_OFFSET]));
        // printf("(1.6) p(phi=1)[%d]: %f\n", action, it->second[0]);
        // printf("(1.7) p(a/phi=1)[%d]: %f\n", action,
        //        it->second[action + ACTION_OFFSET]);
        for (int i = 0; i < 5; i++) {
          printf("#################MIN PROB HACK ACTIVATED#################\n");
        }
        printf("[BEFORE] p(phi): %.10f\n", it->second[0]);
        it->second[0] = (actionMarginals[action] - MIN_PROB) /
                        (it->second[action + ACTION_OFFSET]);
        printf("[AFTER] p(phi): %.10f\n", it->second[0]);
        for (int i = 0; i < 5; i++) {
          printf("####################MIN PROB HACK END####################\n");
        }
        // printf("(1.8) p(phi=1)[%d]: %f\n", action,
        //        it->second[action + ACTION_OFFSET]);
        tmp = log(MIN_PROB) - log(1 - it->second[0]);
        is_min_prob_activated = true;
      }
      sum_log_action_given_phi += tmp;

    } else {
      // Reset the seen features to unseen as its update has already been done.
      it->second[1] = 0;
    }
  }

  return sum_log_action_given_phi;
}

void SarsaEBLearner::exploration_bonus(
    vector<long long>& features,
    long time_step,
    vector<double>& act_exp,
    vector<unordered_map<long long, vector<double>>>& updated_structure) {
  // Calculate p(phi)
  double sum_log_rho_phi = get_sum_log_phi(features, time_step, true);

  // Calculate for all action in Actions p(action/phi)
  vector<double> log_joint_phi_action(numActions);
  double tmp;
  for (int action = 0; action < numActions; action++) {
    tmp =
        get_sum_log_action_given_phi(featureProbs, features, action, time_step);
    log_joint_phi_action[action] = sum_log_rho_phi + tmp;
  }

  // Update the phi values with the currently seen features.
  update_phi(features, time_step);

  // Calculate p'(phi)
  double sum_log_rho_phi_prime = get_sum_log_phi(features, time_step, false);

  double pseudo_count = 0;
  double log_joint_phi_action_prime = 0;
  for (int action = 0; action < numActions; action++) {
    // Copy original Map to tmp Map
    updated_structure[action] = featureProbs;

    // Update the action given phi values for the current action and features.
    update_action_given_phi(updated_structure[action], features, action,
                            time_step);

    // Calculate p'(a,phi)
    tmp = get_sum_log_action_given_phi(updated_structure[action], features,
                                       action, time_step);
    log_joint_phi_action_prime = sum_log_rho_phi_prime + tmp;

    // calculate pseudo count as (1/(exp(p(phi' - p()))-1))
    pseudo_count =
        1.0 /
        (exp(log_joint_phi_action_prime - log_joint_phi_action[action]) - 1);

    // push the exploration bonus to the output vector.
    // printf("log_joint_phi_action_prime: %f\n", log_joint_phi_action_prime);
    // printf("log_joint_phi_action[%d]: %f\n", action,
    //        log_joint_phi_action[action]);
    printf("pseudo_count[%d]: %.20f\n", action, pseudo_count);
    act_exp[action] = beta / sqrt(pseudo_count + 0.01);
  }
}

double SarsaEBLearner::exploration_bonus(vector<long long>& features,
                                         long time_step,
                                         int action) {
  // Calculate p(phi)
  double sum_log_rho_phi = get_sum_log_phi(features, time_step, true);

  // Calculate p(action/phi) for max action
  double sum_log_action_given_phi =
      get_sum_log_action_given_phi(featureProbs, features, action, time_step);
  double log_joint_phi_action = sum_log_rho_phi + sum_log_action_given_phi;

  // Update the phi values with the currently seen features.
  update_phi(features, time_step);

  // Calculate p'(phi)
  double sum_log_rho_phi_prime = get_sum_log_phi(features, time_step, false);

  // Update the action given phi values for the current action and features.
  update_action_given_phi(featureProbs, features, action, time_step);

  // Calculate p'(a,phi)
  sum_log_action_given_phi =
      get_sum_log_action_given_phi(featureProbs, features, action, time_step);
  double log_joint_phi_action_prime =
      sum_log_rho_phi_prime + sum_log_action_given_phi;

  // calculate pseudo count as (1/(exp(p(phi' - p()))-1))
  double pseudo_count =
      1.0 / (exp(log_joint_phi_action_prime - log_joint_phi_action) - 1);

  // push the exploration bonus to the output vector.
  printf("log_joint_phi_action_prime: %f\n", log_joint_phi_action_prime);
  printf("log_joint_phi_action[%d]: %f\n", action, log_joint_phi_action);
  printf("pseudo_count[%d]: %.20f\n", action, pseudo_count);
  return beta / sqrt(pseudo_count + kappa);
}

int SarsaEBLearner::epsilonQI(vector<float>& QValues,
                              vector<float>& QIValues,
                              int episode) {
  randomActionTaken = 0;

  int action = Mathematics::argmax(QValues, agentRand);
  // With probability epsilon: a <- random action in A(s)
  int random = (*agentRand)();
  float epsilon = finalEpsilon;
  if (epsilonDecay && episode <= finalExplorationFrame) {
    epsilon = 1 - (1 - finalEpsilon) * episode / finalExplorationFrame;
  }
  if ((random % int(nearbyint(1.0 / epsilon))) == 0) {
    // if((rand()%int(1.0/epsilon)) == 0){
    randomActionTaken = 1;
    action = Mathematics::argmax(QIValues, agentRand);
  }
  return action;
}

int SarsaEBLearner::optimisticEpsilonQI(vector<float>& QValues,
                                        vector<float>& QIValues,
                                        int episode) {
  randomActionTaken = 0;

  vector<float> mixedQIQValues(QValues.size());

  std::transform(QValues.begin(), QValues.end(), QIValues.begin(),
                 mixedQIQValues.begin(),
                 [&](double q, double qi) { return q + nu * qi; });

  int action = Mathematics::argmax(mixedQIQValues, agentRand);
  // With probability epsilon: a <- random action in A(s)
  int random = (*agentRand)();
  float epsilon = finalEpsilon;
  if (epsilonDecay && episode <= finalExplorationFrame) {
    epsilon = 1 - (1 - finalEpsilon) * episode / finalExplorationFrame;
  }

  if ((random % int(nearbyint(1.0 / epsilon))) == 0) {
    // if((rand()%int(1.0/epsilon)) == 0){
    randomActionTaken = 1;
    vector<float> neqQIValues;
    for (int i = 0; i < QIValues.size(); i++) {
      if (QIValues[i] < 0) {
        neqQIValues.push_back(QIValues[i]);
      }
    }
    if (neqQIValues.size() > 0) {
      printf("Using negative values to take epsilon action\n");
      action = Mathematics::argmax(neqQIValues, agentRand);
    } else {
      action = Mathematics::argmax(QIValues, agentRand);
    }
  }
  return action;
}

void SarsaEBLearner::learnPolicy(ALEInterface& ale, Features* features) {
  struct timeval tvBegin, tvEnd, tvDiff;
  vector<float> reward;
  double elapsedTime;
  double cumReward = 0, prevCumReward = 0;
  sawFirstReward = 0;
  firstReward = 1.0;
  vector<float> episodeResults;
  vector<int> episodeFrames;
  vector<double> episodeFps;

  long time_step = 1;

  long long trueFeatureSize = 0;
  long long trueFnextSize = 0;

  vector<long long> tmp_F;
  double curExpBonus = 0;

  // Initialize action Marginals
  for (int action = 0; action < numActions; action++) {
    actionMarginals[action] = 1.0 / numActions;
  }

  // Repeat (for each episode):
  // This is going to be interrupted by the ALE code since I set max_num_frames
  // beforehand
  for (int episode = episodePassed + 1;
       totalNumberFrames < totalNumberOfFramesToLearn; episode++) {
    // random no-op
    unsigned int noOpNum = 0;
    if (randomNoOp) {
      noOpNum = (*agentRand)() % (noOpMax) + 1;
      for (int i = 0; i < noOpNum; ++i) {
        ale.act(actions[0]);
      }
    }

    // We have to clean the traces every episode:
    for (unsigned int a = 0; a < nonZeroElig.size(); a++) {
      for (unsigned long long i = 0; i < nonZeroElig[a].size(); i++) {
        long long idx = nonZeroElig[a][i];
        e[a][idx] = 0.0;
      }
      nonZeroElig[a].clear();
    }

    F.clear();
    features->getActiveFeaturesIndices(ale.getScreen(), ale.getRAM(), F);
    tmp_F = F;
    trueFeatureSize = F.size();
    groupFeatures(F);
    updateQValues(F, Q);
    updateQIValues(F, QI);

    currentAction = optimisticEpsilonQI(Q, QI, episode);
    // currentAction = epsilonQI(Q, QI, episode);
    // currentAction = epsilonGreedy(Q, episode);
    // currentAction = Mathematics::argmax(Q, agentRand);
    gettimeofday(&tvBegin, NULL);
    int lives = ale.lives();
    // Repeat(for each step of episode) until game is over:
    // This also stops when the maximum number of steps per episode is reached
    while (!ale.game_over()) {
      reward.clear();
      reward.push_back(0.0);
      reward.push_back(0.0);
      updateQValues(F, Q);
      updateQIValues(F, QI);
      updateReplTrace(currentAction, F);

      sanityCheck();
      // Take action, observe reward and next state:
      act(ale, currentAction, reward);
      curExpBonus = exploration_bonus(tmp_F, time_step, currentAction);
      for (int action = 0; action < numActions; action++) {
        printf("Q-value[%d]: %f\n", action, Q[action]);
        printf("QI-value[%d]: %f\n", action, QI[action]);
      }
      cumReward += reward[1];
      if (!ale.game_over()) {
        // Obtain active features in the new state:
        Fnext.clear();
        features->getActiveFeaturesIndices(ale.getScreen(), ale.getRAM(),
                                           Fnext);
        if (Fnext == tmp_F) {
          printf("Both sates are same\n");
        }
        tmp_F = Fnext;
        trueFnextSize = Fnext.size();
        groupFeatures(Fnext);

        // Update Q-values for the new active features
        updateQValues(Fnext, Qnext);
        updateQIValues(Fnext, QInext);

        nextAction = optimisticEpsilonQI(Qnext, QInext, episode);
        // nextAction = epsilonQI(Qnext, QInext, episode);
        // nextAction = epsilonGreedy(Qnext, episode);
        // nextAction = Mathematics::argmax(Qnext, agentRand);
        update_action_marginals(nextAction, time_step);
        printf("reward: %f\n", reward[0]);
        printf("exp_bonus: %.10f\n", curExpBonus);
        printf("action taken: %d\n", currentAction);
      } else {
        nextAction = 0;
        for (unsigned int i = 0; i < Qnext.size(); i++) {
          Qnext[i] = 0;
          QInext[i] = 0;
        }
      }
      // To ensure the learning rate will never increase along
      // the time, Marc used such approach in his JAIR paper
      if (trueFeatureSize > maxFeatVectorNorm) {
        maxFeatVectorNorm = trueFeatureSize;
        learningRate = alpha / maxFeatVectorNorm;
        QI_learningRate = QI_alpha / maxFeatVectorNorm;
      }
      if (!is_min_prob_activated) {
        delta = reward[0] + gamma * Qnext[nextAction] - Q[currentAction];
        QI_delta = curExpBonus + gamma * QInext[nextAction] - QI[currentAction];
      } else {
        delta = 0;
        QI_delta = 0;
        is_min_prob_activated = false;
      }
      printf("delta: %f\n", delta);
      printf("QI_delta: %f\n", QI_delta);
      // Update weights vector:
      for (unsigned int a = 0; a < nonZeroElig.size(); a++) {
        for (unsigned int i = 0; i < nonZeroElig[a].size(); i++) {
          long long idx = nonZeroElig[a][i];
          w[a][idx] = w[a][idx] + learningRate * delta * e[a][idx];
          QI_w[a][idx] = QI_w[a][idx] + QI_learningRate * QI_delta * e[a][idx];
        }
      }
      F = Fnext;
      trueFeatureSize = trueFnextSize;
      currentAction = nextAction;
      time_step++;
    }

    gettimeofday(&tvEnd, NULL);
    timeval_subtract(&tvDiff, &tvEnd, &tvBegin);
    elapsedTime = double(tvDiff.tv_sec) + double(tvDiff.tv_usec) / 1000000.0;

    double fps = double(ale.getEpisodeFrameNumber()) / elapsedTime;
    printf(
        "episode: %d,\t%.0f points,\tavg. return: %.1f,\t%d frames,\t%.0f "
        "fps\n",
        episode, cumReward - prevCumReward, (double)cumReward / (episode),
        ale.getEpisodeFrameNumber(), fps);
    episodeResults.push_back(cumReward - prevCumReward);
    episodeFrames.push_back(ale.getEpisodeFrameNumber());
    episodeFps.push_back(fps);
    totalNumberFrames +=
        ale.getEpisodeFrameNumber() - noOpNum * numStepsPerAction;
    prevCumReward = cumReward;
    features->clearCash();
    ale.reset_game();
    if (toSaveCheckPoint && totalNumberFrames > saveThreshold) {
      saveCheckPoint(episode, totalNumberFrames, episodeResults,
                     saveWeightsEveryXFrames, episodeFrames, episodeFps);
      saveThreshold += saveWeightsEveryXFrames;
    }
  }
}

void SarsaEBLearner::groupFeatures(vector<long long>& activeFeatures) {
  vector<long long> activeGroupIndices;

  int newGroup = 0;
  for (unsigned long long i = 0; i < activeFeatures.size(); ++i) {
    long long featureIndex = activeFeatures[i];
    if (featureTranslate[featureIndex] == 0) {
      if (newGroup) {
        featureTranslate[featureIndex] = numGroups;
        groups[numGroups - 1].numFeatures += 1;
      } else {
        newGroup = 1;
        Group agroup;
        agroup.numFeatures = 1;
        agroup.features.clear();
        groups.push_back(agroup);
        for (unsigned int action = 0; action < w.size(); ++action) {
          w[action].push_back(0.0);
          e[action].push_back(0.0);
          QI_w[action].push_back(init_w_value / activeFeatures.size());
        }
        ++numGroups;
        featureTranslate[featureIndex] = numGroups;
      }
    } else {
      long long groupIndex = featureTranslate[featureIndex] - 1;
      auto it = &groups[groupIndex].features;
      if (it->size() == 0) {
        activeGroupIndices.push_back(groupIndex);
      }
      it->push_back(featureIndex);
    }
  }

  activeFeatures.clear();
  if (newGroup) {
    activeFeatures.push_back(groups.size() - 1);
  }

  for (unsigned long long index = 0; index < activeGroupIndices.size();
       ++index) {
    long long groupIndex = activeGroupIndices[index];
    if (groups[groupIndex].features.size() != groups[groupIndex].numFeatures &&
        groups[groupIndex].features.size() != 0) {
      Group agroup;
      agroup.numFeatures = groups[groupIndex].features.size();
      agroup.features.clear();
      groups.push_back(agroup);
      ++numGroups;
      for (unsigned long long i = 0; i < groups[groupIndex].features.size();
           ++i) {
        featureTranslate[groups[groupIndex].features[i]] = numGroups;
      }
      activeFeatures.push_back(numGroups - 1);
      for (unsigned a = 0; a < w.size(); ++a) {
        w[a].push_back(w[a][groupIndex]);
        e[a].push_back(e[a][groupIndex]);
        QI_w[a].push_back(QI_w[a][groupIndex]);
        if (e[a].back() >= traceThreshold) {
          nonZeroElig[a].push_back(numGroups - 1);
        }
      }
      groups[groupIndex].numFeatures =
          groups[groupIndex].numFeatures - groups[groupIndex].features.size();
    } else if (groups[groupIndex].features.size() ==
               groups[groupIndex].numFeatures) {
      activeFeatures.push_back(groupIndex);
    }
    groups[groupIndex].features.clear();
    groups[groupIndex].features.shrink_to_fit();
  }
}

SarsaEBLearner::~SarsaEBLearner() {}