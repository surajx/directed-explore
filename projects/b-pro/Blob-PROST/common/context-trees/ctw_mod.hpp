#ifndef __CTW_MOD_HPP__
#define __CTW_MOD_HPP__

/******************************
      Author: Joel Veness
        Date: 2011
******************************/

#include "common_mod.hpp"

// boost includes
#include <boost/utility.hpp>
#include <boost/pool/pool.hpp>

// context tree node
class CTNode {
  friend class ContextTree;

 public:
  CTNode();

  /// process a new binary symbol
  void update(bit_t b, bool skip);

  void updateFirst(bit_t b, bool skip, double init_log_prob_kt, long time_step);

  /// log weighted blocked probability
  weight_t logProbWeighted() const;

  /// log KT estimated probability
  weight_t logProbEstimated() const;

  /// child corresponding to a particular symbol
  const CTNode* child(bit_t b) const;

  /// the number of times this context been visited
  int visits() const;

  /// number of descendents
  size_t size() const;

 private:
  // is the current node a leaf node?
  bool isLeaf() const;

  // update the weighted probabilities
  void updateWeighted();

  // compute the logarithm of the KT-estimator update multiplier
  double logKTMul(bit_t b) const;

  weight_t m_log_prob_est;
  weight_t m_log_prob_weighted;

  // one slot for each binary value
  int m_count[2];
  CTNode* m_child[2];
};

// a context tree used for CTW mixing
class ContextTree : public Compressor, boost::noncopyable {
 public:
  /// create a context tree of specified maximum depth and size
  ContextTree(history_t& history, size_t depth, int phase = -1);

  /// delete the context tree
  ~ContextTree();

  /// file extension
  const char* fileExtension() const { return "ctw"; }

  /// the logarithm of the probability of all processed experience
  double logBlockProbability() const;

  // the probability of seeing a particular symbol next
  double prob(bit_t b);

  // the probability of seeing a particular symbol next
  double logProb(bit_t b);

  /// process a new piece of sensory experience
  void update(bit_t b);

  void updateFirst(bit_t b, double init_log_prob_kt, long time_step);

  /// the depth of the context tree
  size_t depth() const;

  /// number of nodes in the context tree
  size_t size() const;

  void initFeatureData(int numActions, long time_step);

  double getP_AGivenPhi_forAction(int action);

  void updateP_AGivenPhi(int numActions, int action, long time_step);

  void setIsChecked(bool is_checked);

  bool getIsChecked();

  struct FeatureData {
    bool is_checked;
    vector<double> p_a_phi;
  };

 private:
  // recover the memory used by a node
  void reclaimMemory(CTNode* n);

  // compute the current context
  void getContext(const history_t& h, context_t& context) const;

  // create (if necessary) all of the nodes in the current context
  void createNodesInCurrentContext(const context_t& context);

  // recursively deletes the nodes in a context tree
  void deleteCT(CTNode* root);

  // not serialized
  boost::pool<> m_ctnode_pool;

  CTNode* m_root;
  int m_phase;
  size_t m_depth;
  history_t& m_history;
  FeatureData featureData;
};

#endif  // __CTW_MOD_HPP__