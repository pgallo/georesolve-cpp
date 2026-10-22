#pragma once

#include <cstddef>
#include <vector>

namespace georesolve {

// Dense matrix, row = previous state, column = next state.
using Matrix = std::vector<std::vector<double>>;

double logSumExp(const std::vector<double>& xs);

// A time-inhomogeneous HMM in log space where the state set may differ at every step.
//   logEmit[t][j]     log b_t(j), |states at t| entries
//   logTrans[t-1][i][j] log a_t(i, j) for t = 1..T-1 (so logTrans.size() == T-1)
//   logInitial[j]     log pi(j); empty means uniform
struct LogHmm {
    std::vector<std::vector<double>> logEmit;
    std::vector<Matrix> logTrans;
    std::vector<double> logInitial;
};

struct ViterbiResult {
    std::vector<std::size_t> path;  // state index per step
    double logScore = 0.0;          // log joint score of the best path
};

ViterbiResult viterbi(const LogHmm& hmm);

// Forward-backward posterior marginals gamma_t(j) = P(state_t = j | all observations).
std::vector<std::vector<double>> posteriors(const LogHmm& hmm);

// log P(observations), the forward log-likelihood.
double logLikelihood(const LogHmm& hmm);

}  // namespace georesolve
