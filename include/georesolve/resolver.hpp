#pragma once

#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "georesolve/corpus.hpp"
#include "georesolve/features.hpp"
#include "georesolve/gazetteer.hpp"
#include "georesolve/knn.hpp"
#include "georesolve/matcher.hpp"
#include "georesolve/transitions.hpp"

namespace georesolve {

enum class Method {
    Population,   // most populous candidate
    Knn,          // highest KNN context vote, population breaks ties
    Independent,  // calibrated feature combination, each mention decided alone
    Hmm,          // calibrated emissions + geographic transitions, Viterbi decoding
};

const char* methodName(Method m);

struct Resolution {
    Mention mention;
    int entry = -1;           // chosen gazetteer index
    double confidence = 0.0;  // posterior probability of the chosen entry
    std::vector<std::pair<int, double>> ranked;  // all candidates, best first
};

struct TrainingReport {
    std::size_t documents = 0;
    std::size_t goldMentions = 0;
    std::size_t alignedMentions = 0;
    std::size_t ambiguousInstances = 0;
    double emissionLogLik = 0.0;
    double transitionWeight = 0.0;
    double transitionScaleKm = 0.0;
    double tuningAccuracy = 0.0;
};

// Pairs gold mentions with matched mentions by order of appearance and surface form.
// Result[i] is the index into `mentions` for gold[i], or -1 when the matcher missed it.
std::vector<int> alignGold(const std::vector<Mention>& mentions, const std::vector<GoldMention>& gold);

class Resolver {
public:
    explicit Resolver(Gazetteer gazetteer, const TransitionParams& params = {});

    // Fits the KNN index, the emission model and (optionally) tunes the transition
    // weight and distance scale on the training documents. Training-time context
    // scores are computed leave-one-document-out so the emission model learns how
    // much to trust KNN on unseen text.
    TrainingReport train(const std::vector<Document>& docs, bool tuneTransitions = true);

    std::vector<Resolution> resolve(std::string_view text, Method method = Method::Hmm) const;

    const Gazetteer& gazetteer() const { return gazetteer_; }
    const EmissionModel& emissionModel() const { return emission_; }
    const TransitionParams& transitionParams() const { return transitions_.params(); }
    double transitionWeight() const { return beta_; }
    void setTransition(const TransitionParams& p, double weight) {
        transitions_ = TransitionModel(gazetteer_, p);
        beta_ = weight;
    }

private:
    struct Prepared {
        std::vector<Token> tokens;
        std::vector<Mention> mentions;
        std::vector<std::vector<double>> knn;
        std::vector<std::vector<FeatureVector>> features;
    };

    Prepared prepare(std::string_view text, int excludeDoc) const;
    std::vector<Resolution> decode(const Prepared& p, Method method) const;
    double trainingAccuracy(const std::vector<Prepared>& prepared, const std::vector<Document>& docs) const;

    Gazetteer gazetteer_;
    MentionMatcher matcher_;
    KnnContextClassifier knn_;
    EmissionModel emission_;
    TransitionModel transitions_;
    double beta_ = 1.0;  // weight of log transitions relative to log emissions
};

}  // namespace georesolve
