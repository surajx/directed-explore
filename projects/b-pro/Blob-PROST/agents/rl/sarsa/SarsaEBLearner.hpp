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
#include "../../../common/context-trees/ctw_mod.hpp"
#endif
#include <vector>
#include <unordered_map>
#include <limits>
#include <math.h>

using namespace std;

class SarsaEBLearner : public SarsaLearner {
 private:
  double beta, sigma, kappa;

  const double nu = 1;
  const double QI_alpha = 0.25;
  double QI_delta;
  vector<float> QI;            // Q(a) entries
  vector<float> QInext;        // Q(a) entries for next action
  vector<vector<float>> QI_w;  // Theta, weights vector
  double MIN_LOG_PROB = 1e-30;
  const long total_feature_count = 114702400;

  // context trees
  unsigned int ct_depth = 4;
  unordered_map<long long, Compressor*> feature_context_trees;

  Compressor* zeroCTPrototype;

  bool is_logging_activated = false;

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

  void initCTForFeature(long long feature);

  void zeroFill(history_t& h, size_t n);

  double ct_exploration_bonus(vector<long long>& features, long time_step);

  void groupFeatures(vector<long long>& activeFeatures);

  int epsilonQI(vector<float>& QValues, vector<float>& QIValues, int episode);

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
