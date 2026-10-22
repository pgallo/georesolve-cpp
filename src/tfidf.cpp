#include "georesolve/tfidf.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <unordered_set>

namespace georesolve {

double dot(const SparseVector& a, const SparseVector& b) {
    double s = 0.0;
    std::size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i].first == b[j].first) {
            s += a[i].second * b[j].second;
            ++i;
            ++j;
        } else if (a[i].first < b[j].first) {
            ++i;
        } else {
            ++j;
        }
    }
    return s;
}

double norm(const SparseVector& a) { return std::sqrt(dot(a, a)); }

double cosine(const SparseVector& a, const SparseVector& b) {
    const double na = norm(a);
    const double nb = norm(b);
    if (na == 0.0 || nb == 0.0) return 0.0;
    return dot(a, b) / (na * nb);
}

void TfIdfVectorizer::fit(const std::vector<std::vector<std::string>>& documents) {
    vocab_.clear();
    std::vector<int> df;
    for (const auto& doc : documents) {
        std::unordered_set<std::string> seen(doc.begin(), doc.end());
        for (const auto& term : seen) {
            auto [it, inserted] = vocab_.emplace(term, static_cast<int>(vocab_.size()));
            if (inserted) df.push_back(0);
            ++df[static_cast<std::size_t>(it->second)];
        }
    }
    const double n = static_cast<double>(documents.size());
    idf_.assign(df.size(), 0.0);
    for (std::size_t t = 0; t < df.size(); ++t) {
        idf_[t] = std::log((1.0 + n) / (1.0 + df[t])) + 1.0;
    }
}

double TfIdfVectorizer::idf(const std::string& term) const {
    auto it = vocab_.find(term);
    return it == vocab_.end() ? 0.0 : idf_[static_cast<std::size_t>(it->second)];
}

SparseVector TfIdfVectorizer::transform(const std::vector<std::string>& terms) const {
    std::map<int, int> counts;  // ordered, so the output is sorted by term id
    for (const auto& term : terms) {
        auto it = vocab_.find(term);
        if (it != vocab_.end()) ++counts[it->second];
    }
    SparseVector v;
    v.reserve(counts.size());
    for (auto [id, tf] : counts) {
        v.emplace_back(id, (1.0 + std::log(static_cast<double>(tf))) * idf_[static_cast<std::size_t>(id)]);
    }
    const double nrm = norm(v);
    if (nrm > 0.0) {
        for (auto& p : v) p.second /= nrm;
    }
    return v;
}

}  // namespace georesolve
