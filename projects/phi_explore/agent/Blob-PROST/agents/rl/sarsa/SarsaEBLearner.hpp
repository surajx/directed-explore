/*******************************************************************************
 **  Implementation of Sarsa(lambda) with Exploration Bonus. It implements Fig.
 **  8.8 (Linear, gradient-descent Sarsa(lambda)) from the book "R. Sutton and
 **  A. Barto; Reinforcement Learning: An Introduction. 1st edition. 1988."
 **  Some updates are made to make it more efficient, as not iterating over all
 **  features.
 **
 ** Author: Suraj Narayanan Sasikumar
 ******************************************************************************/

#ifndef SARSAEBLEARNER_H
#define SARSAEBLEARNER_H
#include "SarsaLearner.hpp"
#endif
#include <vector>
#include <unordered_map>
#include <limits>

using namespace std;

class SarsaEBLearner : public SarsaLearner {
 private:
  double beta, sigma, kappa, init_w_value;
  unordered_map<long long, vector<double>> featureProbs;

  const double nu = 1;
  const double QI_alpha = 0.25;
  double QI_delta;
  float QI_learningRate;
  vector<float> QI;            // Q(a) entries
  vector<float> QInext;        // Q(a) entries for next action
  vector<vector<float>> QI_w;  // Theta, weights vector

  const int ACTION_OFFSET = 2;
  int NUM_PHI_OFFSET;

  bool is_logging_activated = false;
  bool debug_mode = false;

  /**
   * Constructor declared as private to force the user to instantiate
   * SarsaEBLearner informing the parameters to learning/execution.
   */
  SarsaEBLearner();

  //TODO: Documentation
  void add_new_feature_to_map(long long featIdx, int time_step);

  void update_phi(vector<long long>& features, long time_step);

  double get_sum_log_phi(vector<long long>& features,
                         long time_step,
                         bool isFirst);

  double exploration_bonus(vector<long long>& features,
                           long time_step,
                           int action);

  void groupFeatures(vector<long long>& activeFeatures);  

  int epsilonQI(vector<float>& QValues, vector<float>& QIValues, int episode);

  int epsilonQIQ(vector<float>& QValues, vector<float>& QIValues, int episode);

  void saveCheckPoint(int episode, int totalNumberFrames, vector<float>& episodeResults,int& frequency,vector<int>& episodeFrames, vector<double>& episodeFps);
  
  void loadCheckPoint(ifstream& checkPointToLoad);

  void updateQIValues(vector<long long>& Features, vector<float>& QValues);

  int optimisticEpsilonQI(vector<float>& QValues,
                          vector<float>& QIValues,
                          int episode);

  int boltzmannQI(vector<float>& QIvalues, std::mt19937* randAgent);

 public:
  /**
  *   Initialize everything required for SarsaLearner.
  *   Additional params for EB:
  *   @param beta  Exploration rate.
  *   @param sigma Generalization factor.
  */
  SarsaEBLearner(ALEInterface& ale,
                 Features* features,
                 Parameters* param,
                 int seed);

  void learnPolicy(ALEInterface& ale, Features* features);

  ~SarsaEBLearner();
};
