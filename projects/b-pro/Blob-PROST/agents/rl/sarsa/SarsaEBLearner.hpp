/*******************************************************************************
 **  Implementation of Sarsa(lambda) with Exploration Bonus. It implements Fig.
 **  8.8 (Linear, gradient-descent Sarsa(lambda)) from the book "R. Sutton and
 **  A. Barto; Reinforcement Learning: An Introduction. 1st edition. 1988."
 **  Some updates are made to make it more efficient, as not iterating over all
 **  features.
 **
 ** Author:
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
  unordered_map<int, double> actionMarginals;

  const double nu = 1;
  const double QI_alpha = 0.25;
  double QI_delta;
  float QI_learningRate;
  vector<float> QI;            // Q(a) entries
  vector<float> QInext;        // Q(a) entries for next action
  vector<vector<float>> QI_w;  // Theta, weights vector

  bool is_min_prob_activated;

  const int ACTION_OFFSET = 2;
  int NUM_PHI_OFFSET;

  const double MIN_PROB = std::numeric_limits<double>::min();  // 1e-9;

  /**
   * Constructor declared as private to force the user to instantiate
   * SarsaEBLearner informing the parameters to learning/execution.
   */
  SarsaEBLearner();

  /**
  * The distance between current state-action pair and a centroid in feature
  * space.
  * @param vector<long long>& sa_cur a concatenation of the feature vector
  *        actions, obtain simulator screen, RAM, etc.
  * @param vector<long long>& centroid One centroid obtained from a clustering
  *        algorithm. If the number of visited state-action pairs is small,
  *        the centroids may simply be positions of the visited state-action
  *        pairs in feature space.
  */
  // void metric(vector<long long>& sa_cur, vector<long long>& centroid);

  /**
  * Measure of similarity between state-action pairs and a centroid.
  * Computed using a gaussian kernel centered on the centroid. The Measure
  * takes values between 0 and 1.
  * @param vector<long long>& sa_cur A concatenation of the feature vector
  *        and action.
  * @param vector<long long>& centroid One centroid obtained from a clustering
  *        algorithm. If the number of visited state-action pairs is small,
  *        the centroids may simply be positions of the visited state-action
  *        pairs in feature space.
  */
  // void similarity_measure(vector<long long>& sa_cur,
  //                         vector<long long>& centroid);

  /**
  * A generalized state-action visit count. Roughly represents the visit
  * density at a point in feature space.
  * @param vector<long long>& sa_cur A concatenation of the feature vector
  *        and action.
  */
  // double pseudo_count(vector<long long>& sa_cur);

  /**
  * A bonus added to the external reward to incentivise exploration. Roughly
  * a local information reward for visiting a state-action pair.
  * @param vector<long long>& features The feature used in the linear approx.
  * of the Q-function.
  * @param long int: time steps elapsed
  */
  // double exploration_bonus(vector<long long>& features, long time_step);

  /**
  * Update the running probablity measure for occurance for each feature.
  * Update Formula: mu_{t+1} = mu_t + (phi_{t+1} - mu_t)/(1+t)
  * @param vector<long long>& features The feature used in the linear approx.
  * of the Q-function.
  * @param int no of time steps elapsed.
  */
  // void update_prob_feature(vector<long long>& features, long time_step);

  /**
  * If the feature has not been seen before we create a new map entry of the
  *   form [feature index] : vector {
  *              p(phi_{i}),
  *              seen_flag, # 1-seen, 0-not
  *              p(a_1/phi_{i}),...,p(a_n/phi_{i}),
  *              n_phi # No of time phi has been active
  *   }
  */
  void add_new_feature_to_map(long long featIdx, int time_step);

  void update_action_marginals(int cur_action, int time_step);

  void update_phi(vector<long long>& features, long time_step);

  void update_action_given_phi(
      unordered_map<long long, vector<double>>& tmp_featureProbs,
      vector<long long>& features,
      int action,
      long time_step);

  double get_sum_log_phi(vector<long long>& features,
                         long time_step,
                         bool isFirst);

  double get_sum_log_action_given_phi(
      unordered_map<long long, vector<double>>& context_featureProbs,
      vector<long long>& features,
      int action,
      long time_step);

  void exploration_bonus(
      vector<long long>& features,
      long time_step,
      vector<double>& act_exp,
      vector<unordered_map<long long, vector<double>>>& updated_structure);

  double exploration_bonus(vector<long long>& features,
                           long time_step,
                           int action);

  void groupFeatures(vector<long long>& activeFeatures);

  int epsilonQI(vector<float>& QValues, vector<float>& QIValues, int episode);

  void updateQIValues(vector<long long>& Features, vector<float>& QValues);

  int optimisticEpsilonQI(vector<float>& QValues,
                          vector<float>& QIValues,
                          int episode);

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
