#pragma once

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace georesolve {

// Sparse vector sorted by term id.
using SparseVector = std::vector<std::pair<int, double>>;

double dot(const SparseVector& a, const SparseVector& b);
double norm(const SparseVector& a);
double cosine(const SparseVector& a, const SparseVector& b);

// TF-IDF with smoothed idf: idf(t) = ln((1 + N) / (1 + df(t))) + 1, sublinear tf
// (1 + ln tf), L2-normalized output. Terms unseen at fit time are dropped.
class TfIdfVectorizer {
public:
    void fit(const std::vector<std::vector<std::string>>& documents);
    SparseVector transform(const std::vector<std::string>& terms) const;

    std::size_t vocabularySize() const { return vocab_.size(); }
    double idf(const std::string& term) const;  // 0 for unknown terms

private:
    std::unordered_map<std::string, int> vocab_;
    std::vector<double> idf_;
};

}  // namespace georesolve
