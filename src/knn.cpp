#include "georesolve/knn.hpp"

#include <algorithm>

namespace georesolve {

void KnnContextClassifier::fit(std::vector<LabelledContext> examples) {
    std::vector<std::vector<std::string>> docs;
    docs.reserve(examples.size());
    for (const auto& e : examples) docs.push_back(e.terms);
    tfidf_.fit(docs);

    examples_.clear();
    byLabel_.clear();
    for (auto& e : examples) {
        examples_.push_back({e.docId, e.label, tfidf_.transform(e.terms)});
        byLabel_[e.label].push_back(examples_.size() - 1);
    }
}

std::vector<double> KnnContextClassifier::predict(const SparseVector& query, const std::vector<int>& candidates,
                                                  int excludeDoc) const {
    const std::size_t n = candidates.size();
    std::vector<double> probs(n, n == 0 ? 0.0 : 1.0 / static_cast<double>(n));
    if (n == 0 || query.empty()) return probs;

    struct Neighbour {
        double sim;
        std::size_t candidate;
    };
    std::vector<Neighbour> neighbours;
    for (std::size_t c = 0; c < n; ++c) {
        auto it = byLabel_.find(candidates[c]);
        if (it == byLabel_.end()) continue;
        for (std::size_t idx : it->second) {
            const Stored& s = examples_[idx];
            if (s.docId == excludeDoc) continue;
            double sim = dot(query, s.vec);  // both sides are L2-normalized
            if (sim > 0.0) neighbours.push_back({sim, c});
        }
    }
    if (neighbours.empty()) return probs;

    const std::size_t k = std::min<std::size_t>(static_cast<std::size_t>(k_), neighbours.size());
    std::partial_sort(neighbours.begin(), neighbours.begin() + static_cast<std::ptrdiff_t>(k), neighbours.end(),
                      [](const Neighbour& a, const Neighbour& b) {
                          if (a.sim != b.sim) return a.sim > b.sim;
                          return a.candidate < b.candidate;  // deterministic tie-break
                      });

    std::vector<double> votes(n, 0.0);
    double total = 0.0;
    for (std::size_t i = 0; i < k; ++i) {
        votes[neighbours[i].candidate] += neighbours[i].sim;
        total += neighbours[i].sim;
    }
    const double denom = total + smoothing_ * static_cast<double>(n);
    for (std::size_t c = 0; c < n; ++c) probs[c] = (votes[c] + smoothing_) / denom;
    return probs;
}

}  // namespace georesolve
