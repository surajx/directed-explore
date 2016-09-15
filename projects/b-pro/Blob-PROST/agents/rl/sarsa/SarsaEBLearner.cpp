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
#include "../../../common/context-trees/ctw_mod.hpp"
#include "../../../common/context-trees/common_mod.hpp"

#include <boost/program_options.hpp>
#include <boost/progress.hpp>
#include <boost/filesystem/operations.hpp>

#endif
#include "SarsaEBLearner.hpp"
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <set>
#include <random>

#include <algorithm>

using namespace std;
namespace po = boost::program_options;

SarsaEBLearner::SarsaEBLearner(ALEInterface& ale,
                               Features* features,
                               Parameters* param,
                               int seed)
    : SarsaLearner(ale, features, param, seed) {
  printf("SarsaEBLearner is Running the show!!!\n");
  beta = param->getBeta();
  sigma = param->getSigma();
  kappa = param->getKappa();

  // init_w_value = beta / (sqrt(kappa) * (1 - gamma));
  // TODO: talk about initialization
  // init_w_value = beta / sqrt(kappa);

  for (int i = 0; i < numActions; i++) {
    // Initialize Q;
    QI.push_back(0);
    QInext.push_back(0);
    // Initialize w:
    QI_w.push_back(vector<float>());
  }
}

void SarsaEBLearner::initCTForFeature(long long feature) {
  Compressor* ct = new ContextTree(ct_depth);
  feature_context_trees[feature] = ct;
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

double SarsaEBLearner::ct_exploration_bonus(vector<long long>& features,
                                            long time_step) {
  // For each features get logProb from context tree and take sum to get the
  // joint.
  double sum_log_phi = 0;
  double sum_log_phi_prime = 0;
  for (long long featIdx : features) {
    if (feature_context_trees.find(featIdx) == feature_context_trees.end()) {
      initCTForFeature(featIdx);
      sum_log_phi += zeroCTPrototype->logProb(1);

      double root_prob = zeroCTPrototype->getRootLogProbEstimated();
      feature_context_trees[featIdx]->updateFirst(1, root_prob, time_step);
    } else {
      sum_log_phi += feature_context_trees[featIdx]->logProb(1);
      feature_context_trees[featIdx]->update(1);
    }

    // p(phi_i=1)
    feature_context_trees[featIdx]->setFeatureChecked(true);
    sum_log_phi_prime += feature_context_trees[featIdx]->logProb(1);
  }

  // Add contribution of implicit P(prototype = 0) trees
  sum_log_phi += zeroCTPrototype->logProb(0);
  zeroCTPrototype->update(0);
  sum_log_phi_prime += zeroCTPrototype->logProb(0);

  // p(phi_i=0)
  for (auto it = feature_context_trees.begin();
       it != feature_context_trees.end(); ++it) {
    if (!it->second->isFeatureChecked()) {
      sum_log_phi += it->second->logProb(0);
      it->second->update(0);
      sum_log_phi_prime += it->second->logProb(0);
    } else {
      // Reset the seen features to unseen as its update has already been
      // done.
      it->second->setFeatureChecked(false);
    }
  }

  double pseudo_count;
  bool is_rho_equal_rho_prime = ((sum_log_phi_prime - sum_log_phi) < DBL_MIN);
  bool is_prob_overflow =
      (sum_log_phi_prime < -DBL_MAX) || (sum_log_phi < -DBL_MAX);

  std::cout << "sum_log_phi: " << sum_log_phi << std::endl;
  std::cout << "sum_log_phi_prime: " << sum_log_phi_prime << std::endl;
  std::cout << "rho_equals_rho_prime: " << is_rho_equal_rho_prime << std::endl;

  if (is_rho_equal_rho_prime) {
    return 0;
  }
  if (is_prob_overflow) {
    return beta / sqrt(kappa);
  }
  // } else if ((sum_log_phi_prime - sum_log_phi) < 100) {
  //   pseudo_count = exp(log(1 - exp(sum_log_phi_prime)) -
  //                      log(exp(sum_log_phi_prime - sum_log_phi) - 1));
  pseudo_count = 1 / (exp(sum_log_phi_prime - sum_log_phi) - 1);

  if (enable_logging) {
    printf("sum_log_phi: %f\n", sum_log_phi);
    printf("sum_log_phi_prime: %f\n", sum_log_phi_prime);
    printf("pseudo count: %.15f\n", pseudo_count);
  }

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
    action = boltzmannQI(QIValues, agentRand);
  }
  return action;
}

int SarsaEBLearner::boltzmannQI(vector<float>& QIvalues,
                                std::mt19937* randAgent) {
  double max = QIvalues[0];
  double min = QIvalues[0];
  vector<double> weights(QIvalues.size());
  for (float q : QIvalues) {
    if (max < q)
      max = q;
    if (min > q)
      min = q;
  }
  double tau = pow(fabs(max - min) + 1, 2);

  for (int idx = 0; idx < QIvalues.size(); idx++) {
    weights[idx] = exp(QIvalues[idx] / tau);
  }

  // printf("maxQI: %f\n", max);
  // printf("minQI: %f\n", min);
  // printf("QIVal size: %d\n", QIvalues.size());

  std::discrete_distribution<int> boltDist(weights.begin(), weights.end());

  int action = boltDist(*agentRand);
  randomActionTaken = 1;

  if (enable_logging) {
    printf("tau: %f\n", tau);
    printf("Random action: %d\n", action);
  }

  return action;
}

void SarsaEBLearner::learnPolicy(ALEInterface& ale, Features* features) {
  // initialize zeroCTPrototype Context Tree
  zeroCTPrototype = new ContextTree(ct_depth);

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

  int logging_count = 1;
  int logging_ep_start = 1;

  // vector<long long> tmp_F;
  double curExpBonus = 0;
  // Repeat (for each episode):
  // This is going to be interrupted by the ALE code since I set
  // max_num_framesupdate
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

    if (!enable_logging && totalNumberFrames >= 200000 * logging_count) {
      enable_logging = true;
      logging_count++;
      logging_ep_start = episode;
    }

    if (episode >= logging_ep_start + 5) {
      enable_logging = false;
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
      cumReward += reward[1];
      if (!ale.game_over()) {
        // Obtain active features in the new state:
        Fnext.clear();
        features->getActiveFeaturesIndices(ale.getScreen(), ale.getRAM(),
                                           Fnext);
        curExpBonus = ct_exploration_bonus(Fnext, time_step);
        trueFnextSize = Fnext.size();
        groupFeatures(Fnext);

        // Update Q-values for the new active features
        updateQValues(Fnext, Qnext);
        updateQIValues(Fnext, QInext);

        nextAction = optimisticEpsilonQI(Qnext, QInext, episode);
        // nextAction = epsilonQI(Qnext, QInext, episode);
        // nextAction = epsilonGreedy(Qnext, episode);
        // nextAction = Mathematics::argmax(Qnext, agentRand);
        if (enable_logging) {
          for (int action = 0; action < numActions; action++) {
            printf("Q-value[%d]: %f\n", action, Q[action]);
            printf("QI-value[%d]: %f\n", action, QI[action]);
          }
          printf("reward: %f\n", reward[0]);
          printf("exp_bonus: %.10f\n", curExpBonus);
          printf("action taken: %d\n", currentAction);
        }
      } else {
        nextAction = 0;
        int missedSteps = episodeLength - ale.getEpisodeFrameNumber() + 1;
        double penalty = pow(gamma, missedSteps) - 1;
        curExpBonus -= penalty;
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
      }

      // optimistic scaling of exploration bonus
      curExpBonus += gamma - 1.0;

      delta = reward[0] + gamma * Qnext[nextAction] - Q[currentAction];
      QI_delta = curExpBonus + gamma * QInext[nextAction] - QI[currentAction];
      if (enable_logging) {
        printf("delta: %f\n", delta);
        printf("QI_delta: %f\n", QI_delta);
      }
      // Update weights vector:
      for (unsigned int a = 0; a < nonZeroElig.size(); a++) {
        for (unsigned int i = 0; i < nonZeroElig[a].size(); i++) {
          long long idx = nonZeroElig[a][i];
          w[a][idx] = w[a][idx] + learningRate * delta * e[a][idx];
          QI_w[a][idx] = QI_w[a][idx] + learningRate * QI_delta * e[a][idx];
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
          QI_w[action].push_back(0.0);
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