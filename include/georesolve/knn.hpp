#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "georesolve/tfidf.hpp"

namespace georesolve {

struct LabelledContext {
    int docId = -1;  // source document, used for leave-one-document-out scoring
    int label = -1;  // gazetteer index of the gold entry
    std::vector<std::string> terms;
};

// Instance-based context classifier. Every labelled training mention becomes a
// TF-IDF vector of its sentence context; a query mention is compared by cosine
// to the training contexts of its own candidates only, and the k most similar
// contexts vote with weight equal to their similarity.
class KnnContextClassifier {
public:
    explicit KnnContextClassifier(int k = 5, double smoothing = 0.25) : k_(k), smoothing_(smoothing) {}

    void fit(std::vector<LabelledContext> examples);

    SparseVector vectorize(const std::vector<std::string>& terms) const { return tfidf_.transform(terms); }

    // Smoothed vote distribution over `candidates` (same order):
    //   p(c) = (votes(c) + s) / (sum_c' votes(c') + s * |C|)
    // Examples whose docId equals excludeDoc are ignored. When no candidate has a
    // similar training context the result is uniform.
    std::vector<double> predict(const SparseVector& query, const std::vector<int>& candidates,
                                int excludeDoc = -1) const;

    std::size_t exampleCount() const { return examples_.size(); }
    const TfIdfVectorizer& vectorizer() const { return tfidf_; }

private:
    struct Stored {
        int docId;
        int label;
        SparseVector vec;
    };
    int k_;
    double smoothing_;
    TfIdfVectorizer tfidf_;
    std::vector<Stored> examples_;
    std::unordered_map<int, std::vector<std::size_t>> byLabel_;
};

}  // namespace georesolve
