#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "georesolve/gazetteer.hpp"
#include "georesolve/matcher.hpp"
#include "georesolve/tokenizer.hpp"

namespace georesolve {

enum Feature : std::size_t {
    kLogPopulation = 0,   // log10(pop+1) relative to the most populous candidate (<= 0)
    kLogKnn,              // log of the KNN context vote share
    kQualifiedBy,         // "Paris, Texas": the next mention is an area containing this candidate
    kQualifies,           // "Texas" in "Paris, Texas": this area contains a candidate of the previous mention
    kAnchorContains,      // an unambiguous mention elsewhere is an area containing this candidate
    kContainsAnchor,      // this candidate is an area containing an unambiguous mention
    kAnchorSameCountry,   // an unambiguous mention shares this candidate's country
    kAnchorProximity,     // -log10(1 + d/50km) to the nearest unambiguous point mention, 0 if none
    kIsArea,              // candidate is a country or first-level subdivision
    kNumFeatures
};

using FeatureVector = std::array<double, kNumFeatures>;

const std::array<const char*, kNumFeatures>& featureNames();

// Computes one feature vector per candidate per mention. knnProbs[m][c] must be
// aligned with mentions[m].candidates[c].
std::vector<std::vector<FeatureVector>> extractFeatures(const Gazetteer& gazetteer,
                                                        const std::vector<Token>& tokens,
                                                        const std::vector<Mention>& mentions,
                                                        const std::vector<std::vector<double>>& knnProbs);

// Multinomial conditional logit over a mention's candidates:
//   P(c | mention) = exp(w . f(c)) / sum_c' exp(w . f(c'))
// Trained by full-batch gradient ascent on the log-likelihood of the gold
// candidate with an L2 penalty. The softmax output is the calibrated emission
// score used by the HMM.
class EmissionModel {
public:
    struct Instance {
        std::vector<FeatureVector> candidates;
        std::size_t gold = 0;
    };

    EmissionModel();

    // Returns the final mean log-likelihood on the training instances.
    double fit(const std::vector<Instance>& instances, int iterations = 3000, double learningRate = 0.5,
               double l2 = 1e-3);

    std::vector<double> logProbs(const std::vector<FeatureVector>& candidates) const;

    const FeatureVector& weights() const { return w_; }
    void setWeights(const FeatureVector& w) { w_ = w; }

private:
    FeatureVector w_{};
};

}  // namespace georesolve
