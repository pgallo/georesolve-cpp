#include "georesolve/transitions.hpp"

#include <cmath>
#include <stdexcept>

#include "georesolve/geo.hpp"

namespace georesolve {

double coherence(const Entry& a, const Entry& b, const TransitionParams& p) {
    double coh = -p.lambda * std::log1p(haversineKm(a.lat, a.lon, b.lat, b.lon) / p.scaleKm);
    if (a.country == b.country) coh += p.sameCountry;
    if (Gazetteer::contains(a, b) || Gazetteer::contains(b, a)) coh += p.containment;
    return coh;
}

TransitionModel::TransitionModel(const Gazetteer& gazetteer, const TransitionParams& params)
    : params_(params) {
    const auto& entries = gazetteer.entries();
    logZ_.assign(entries.size(), 0.0);
    if (params.normalization != Normalization::Gazetteer) return;
    std::vector<double> row(entries.size());
    for (std::size_t i = 0; i < entries.size(); ++i) {
        for (std::size_t j = 0; j < entries.size(); ++j) row[j] = coherence(entries[i], entries[j], params);
        logZ_[i] = logSumExp(row);
    }
}

Matrix TransitionModel::logMatrix(const Gazetteer& g, const std::vector<int>& prev, const std::vector<int>& cur) const {
    if (logZ_.size() != g.size()) throw std::logic_error("TransitionModel built for a different gazetteer");
    Matrix m(prev.size(), std::vector<double>(cur.size(), 0.0));
    for (std::size_t i = 0; i < prev.size(); ++i) {
        const Entry& a = g.at(prev[i]);
        const double lz = logNormalizer(prev[i]);
        for (std::size_t j = 0; j < cur.size(); ++j) m[i][j] = coherence(a, g.at(cur[j]), params_) - lz;
        if (params_.normalization == Normalization::Local) {
            const double z = logSumExp(m[i]);
            for (double& v : m[i]) v -= z;
        }
    }
    return m;
}

}  // namespace georesolve
