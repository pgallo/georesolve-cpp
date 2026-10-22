#include "georesolve/features.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "georesolve/geo.hpp"
#include "georesolve/hmm.hpp"

namespace georesolve {

const std::array<const char*, kNumFeatures>& featureNames() {
    static const std::array<const char*, kNumFeatures> names = {
        "log_population", "log_knn",         "qualified_by",        "qualifies",  "anchor_contains",
        "contains_anchor", "anchor_same_country", "anchor_proximity", "is_area",
    };
    return names;
}

namespace {

// True when mention b directly follows mention a as a qualifier: "A, B" or "A in B".
bool isQualifierPair(const std::vector<Token>& tokens, const Mention& a, const Mention& b) {
    if (b.tokenBegin == a.tokenEnd) return tokens[b.tokenBegin].punctBefore == ',';
    if (b.tokenBegin == a.tokenEnd + 1) {
        const Token& between = tokens[a.tokenEnd];
        return between.norm == "in" && between.punctBefore == 0 && tokens[b.tokenBegin].punctBefore == 0;
    }
    return false;
}

bool anyContains(const Gazetteer& g, const std::vector<int>& areas, const Entry& place) {
    return std::any_of(areas.begin(), areas.end(),
                       [&](int a) { return Gazetteer::contains(g.at(a), place); });
}

}  // namespace

std::vector<std::vector<FeatureVector>> extractFeatures(const Gazetteer& g, const std::vector<Token>& tokens,
                                                        const std::vector<Mention>& mentions,
                                                        const std::vector<std::vector<double>>& knnProbs) {
    // Unambiguous mentions act as anchors for every other mention in the text.
    std::vector<std::pair<std::size_t, int>> anchors;  // (mention index, entry)
    for (std::size_t m = 0; m < mentions.size(); ++m) {
        if (mentions[m].candidates.size() == 1) anchors.emplace_back(m, mentions[m].candidates[0]);
    }

    std::vector<std::vector<FeatureVector>> out(mentions.size());
    for (std::size_t m = 0; m < mentions.size(); ++m) {
        const Mention& mention = mentions[m];
        double maxLogPop = -std::numeric_limits<double>::infinity();
        for (int c : mention.candidates) {
            maxLogPop = std::max(maxLogPop, std::log10(static_cast<double>(g.at(c).population) + 1.0));
        }
        const Mention* next = (m + 1 < mentions.size() && isQualifierPair(tokens, mention, mentions[m + 1]))
                                  ? &mentions[m + 1]
                                  : nullptr;
        const Mention* prev = (m > 0 && isQualifierPair(tokens, mentions[m - 1], mention)) ? &mentions[m - 1]
                                                                                            : nullptr;

        for (std::size_t ci = 0; ci < mention.candidates.size(); ++ci) {
            const Entry& e = g.at(mention.candidates[ci]);
            FeatureVector f{};
            f[kLogPopulation] = std::log10(static_cast<double>(e.population) + 1.0) - maxLogPop;
            f[kLogKnn] = std::log(std::max(knnProbs[m][ci], 1e-9));
            f[kQualifiedBy] = (next != nullptr && anyContains(g, next->candidates, e)) ? 1.0 : 0.0;
            if (prev != nullptr && e.isAdminArea()) {
                for (int p : prev->candidates) {
                    if (Gazetteer::contains(e, g.at(p))) f[kQualifies] = 1.0;
                }
            }
            double nearest = std::numeric_limits<double>::infinity();
            for (const auto& [am, aEntry] : anchors) {
                if (am == m) continue;
                const Entry& a = g.at(aEntry);
                if (Gazetteer::contains(a, e)) f[kAnchorContains] = 1.0;
                if (Gazetteer::contains(e, a)) f[kContainsAnchor] = 1.0;
                if (a.country == e.country) f[kAnchorSameCountry] = 1.0;
                if (!a.isAdminArea()) nearest = std::min(nearest, haversineKm(a.lat, a.lon, e.lat, e.lon));
            }
            if (std::isfinite(nearest)) f[kAnchorProximity] = -std::log10(1.0 + nearest / 50.0);
            f[kIsArea] = e.isAdminArea() ? 1.0 : 0.0;
            out[m].push_back(f);
        }
    }
    return out;
}

EmissionModel::EmissionModel() {
    // Untrained default: population prior plus context votes, equally weighted.
    w_[kLogPopulation] = 1.0;
    w_[kLogKnn] = 1.0;
}

std::vector<double> EmissionModel::logProbs(const std::vector<FeatureVector>& candidates) const {
    std::vector<double> scores;
    scores.reserve(candidates.size());
    for (const auto& f : candidates) {
        double s = 0.0;
        for (std::size_t k = 0; k < kNumFeatures; ++k) s += w_[k] * f[k];
        scores.push_back(s);
    }
    const double z = logSumExp(scores);
    for (auto& s : scores) s -= z;
    return scores;
}

double EmissionModel::fit(const std::vector<Instance>& instances, int iterations, double learningRate, double l2) {
    if (instances.empty()) return 0.0;
    const double n = static_cast<double>(instances.size());
    double meanLogLik = 0.0;
    for (int it = 0; it < iterations; ++it) {
        FeatureVector grad{};
        meanLogLik = 0.0;
        for (const auto& inst : instances) {
            auto lp = logProbs(inst.candidates);
            meanLogLik += lp[inst.gold] / n;
            // d/dw log P(gold) = f(gold) - E_p[f]
            for (std::size_t c = 0; c < inst.candidates.size(); ++c) {
                const double p = std::exp(lp[c]);
                const double indicator = (c == inst.gold) ? 1.0 : 0.0;
                for (std::size_t k = 0; k < kNumFeatures; ++k) {
                    grad[k] += (indicator - p) * inst.candidates[c][k] / n;
                }
            }
        }
        for (std::size_t k = 0; k < kNumFeatures; ++k) w_[k] += learningRate * (grad[k] - l2 * w_[k]);
    }
    return meanLogLik;
}

}  // namespace georesolve
