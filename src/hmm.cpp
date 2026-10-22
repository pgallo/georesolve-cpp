#include "georesolve/hmm.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace georesolve {

namespace {

constexpr double kNegInf = -std::numeric_limits<double>::infinity();

void validate(const LogHmm& hmm) {
    const std::size_t T = hmm.logEmit.size();
    if (T == 0) return;
    if (hmm.logTrans.size() != T - 1) throw std::invalid_argument("hmm: need T-1 transition matrices");
    if (!hmm.logInitial.empty() && hmm.logInitial.size() != hmm.logEmit[0].size()) {
        throw std::invalid_argument("hmm: initial distribution size mismatch");
    }
    for (std::size_t t = 1; t < T; ++t) {
        const Matrix& a = hmm.logTrans[t - 1];
        if (a.size() != hmm.logEmit[t - 1].size()) throw std::invalid_argument("hmm: transition rows mismatch");
        for (const auto& row : a) {
            if (row.size() != hmm.logEmit[t].size()) throw std::invalid_argument("hmm: transition cols mismatch");
        }
    }
}

double initial(const LogHmm& hmm, std::size_t j) {
    if (!hmm.logInitial.empty()) return hmm.logInitial[j];
    return -std::log(static_cast<double>(hmm.logEmit[0].size()));
}

std::vector<std::vector<double>> forward(const LogHmm& hmm) {
    const std::size_t T = hmm.logEmit.size();
    std::vector<std::vector<double>> alpha(T);
    alpha[0].resize(hmm.logEmit[0].size());
    for (std::size_t j = 0; j < alpha[0].size(); ++j) alpha[0][j] = initial(hmm, j) + hmm.logEmit[0][j];
    std::vector<double> terms;
    for (std::size_t t = 1; t < T; ++t) {
        const Matrix& a = hmm.logTrans[t - 1];
        alpha[t].resize(hmm.logEmit[t].size());
        for (std::size_t j = 0; j < alpha[t].size(); ++j) {
            terms.clear();
            for (std::size_t i = 0; i < alpha[t - 1].size(); ++i) terms.push_back(alpha[t - 1][i] + a[i][j]);
            alpha[t][j] = logSumExp(terms) + hmm.logEmit[t][j];
        }
    }
    return alpha;
}

}  // namespace

double logSumExp(const std::vector<double>& xs) {
    if (xs.empty()) return kNegInf;
    const double m = *std::max_element(xs.begin(), xs.end());
    if (!std::isfinite(m)) return m;
    double s = 0.0;
    for (double x : xs) s += std::exp(x - m);
    return m + std::log(s);
}

ViterbiResult viterbi(const LogHmm& hmm) {
    validate(hmm);
    ViterbiResult result;
    const std::size_t T = hmm.logEmit.size();
    if (T == 0) return result;

    // delta_t(j) = max_i [delta_{t-1}(i) + log a_t(i,j)] + log b_t(j)
    std::vector<double> delta(hmm.logEmit[0].size());
    for (std::size_t j = 0; j < delta.size(); ++j) delta[j] = initial(hmm, j) + hmm.logEmit[0][j];
    std::vector<std::vector<std::size_t>> backptr(T);

    for (std::size_t t = 1; t < T; ++t) {
        const Matrix& a = hmm.logTrans[t - 1];
        std::vector<double> next(hmm.logEmit[t].size(), kNegInf);
        backptr[t].assign(next.size(), 0);
        for (std::size_t j = 0; j < next.size(); ++j) {
            for (std::size_t i = 0; i < delta.size(); ++i) {
                const double s = delta[i] + a[i][j];
                if (s > next[j]) {
                    next[j] = s;
                    backptr[t][j] = i;
                }
            }
            next[j] += hmm.logEmit[t][j];
        }
        delta = std::move(next);
    }

    std::size_t best = static_cast<std::size_t>(std::max_element(delta.begin(), delta.end()) - delta.begin());
    result.logScore = delta[best];
    result.path.assign(T, 0);
    result.path[T - 1] = best;
    for (std::size_t t = T - 1; t > 0; --t) result.path[t - 1] = backptr[t][result.path[t]];
    return result;
}

std::vector<std::vector<double>> posteriors(const LogHmm& hmm) {
    validate(hmm);
    const std::size_t T = hmm.logEmit.size();
    if (T == 0) return {};
    auto alpha = forward(hmm);

    std::vector<std::vector<double>> beta(T);
    beta[T - 1].assign(hmm.logEmit[T - 1].size(), 0.0);
    std::vector<double> terms;
    for (std::size_t t = T - 1; t > 0; --t) {
        const Matrix& a = hmm.logTrans[t - 1];
        beta[t - 1].resize(hmm.logEmit[t - 1].size());
        for (std::size_t i = 0; i < beta[t - 1].size(); ++i) {
            terms.clear();
            for (std::size_t j = 0; j < beta[t].size(); ++j) {
                terms.push_back(a[i][j] + hmm.logEmit[t][j] + beta[t][j]);
            }
            beta[t - 1][i] = logSumExp(terms);
        }
    }

    const double logZ = logSumExp(alpha[T - 1]);
    std::vector<std::vector<double>> gamma(T);
    for (std::size_t t = 0; t < T; ++t) {
        gamma[t].resize(alpha[t].size());
        for (std::size_t j = 0; j < gamma[t].size(); ++j) gamma[t][j] = std::exp(alpha[t][j] + beta[t][j] - logZ);
    }
    return gamma;
}

double logLikelihood(const LogHmm& hmm) {
    validate(hmm);
    if (hmm.logEmit.empty()) return 0.0;
    return logSumExp(forward(hmm).back());
}

}  // namespace georesolve
