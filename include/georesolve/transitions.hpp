#pragma once

#include <vector>

#include "georesolve/gazetteer.hpp"
#include "georesolve/hmm.hpp"

namespace georesolve {

// Geographic coherence between consecutive mentions:
//   coh(i, j) = -lambda * ln(1 + d(i, j) / scaleKm)
//               + sameCountry * [cc(i) == cc(j)]
//               + containment * [i contains j or j contains i]
enum class Normalization {
    Local,      // over the next mention's candidates only
    Gazetteer,  // over every gazetteer entry (a proper first-order Markov chain on places)
    None,       // raw potential; the chain becomes a linear-chain random field
};

struct TransitionParams {
    Normalization normalization = Normalization::Gazetteer;
    double scaleKm = 100.0;
    double lambda = 1.0;
    double sameCountry = 1.0;
    double containment = 2.0;
};

double coherence(const Entry& a, const Entry& b, const TransitionParams& params);

// Transition model whose state space is the whole gazetteer:
//   log a(i, j) = coh(i, j) - log Z(i),   Z(i) = sum over every entry j' of exp(coh(i, j'))
// Normalizing over the full gazetteer (not just the next mention's candidates)
// keeps the absolute coherence signal: a pair of distant candidates is penalized
// even if every alternative for the next mention is also distant.
class TransitionModel {
public:
    TransitionModel() = default;
    TransitionModel(const Gazetteer& gazetteer, const TransitionParams& params);

    const TransitionParams& params() const { return params_; }
    double logNormalizer(int entry) const { return logZ_.at(static_cast<std::size_t>(entry)); }

    // `gazetteer` must be the one the model was built from.
    Matrix logMatrix(const Gazetteer& gazetteer, const std::vector<int>& prev, const std::vector<int>& cur) const;

private:
    TransitionParams params_;
    std::vector<double> logZ_;
};

}  // namespace georesolve
