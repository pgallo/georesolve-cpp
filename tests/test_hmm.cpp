#include <gtest/gtest.h>

#include <cmath>

#include "georesolve/geo.hpp"
#include "georesolve/hmm.hpp"
#include "georesolve/transitions.hpp"

using namespace georesolve;

namespace {

std::vector<double> logv(std::vector<double> v) {
    for (double& x : v) x = std::log(x);
    return v;
}

Matrix logm(Matrix m) {
    for (auto& row : m) row = logv(row);
    return m;
}

// Two steps, two states each, uniform start.
//   b_0 = [0.9, 0.1], b_1 = [0.2, 0.8], A = [[0.7, 0.3], [0.4, 0.6]]
// Joint path scores (pi * b0 * a * b1):
//   (0,0) = .5*.9*.7*.2 = 0.063
//   (0,1) = .5*.9*.3*.8 = 0.108   <- best
//   (1,0) = .5*.1*.4*.2 = 0.004
//   (1,1) = .5*.1*.6*.8 = 0.024
//   total = 0.199
LogHmm toy() {
    LogHmm h;
    h.logEmit = {logv({0.9, 0.1}), logv({0.2, 0.8})};
    h.logTrans = {logm({{0.7, 0.3}, {0.4, 0.6}})};
    return h;
}

}  // namespace

TEST(Viterbi, HandComputedTwoStepExample) {
    auto r = viterbi(toy());
    EXPECT_EQ(r.path, (std::vector<std::size_t>{0, 1}));
    EXPECT_NEAR(r.logScore, std::log(0.108), 1e-12);
}

TEST(ForwardBackward, HandComputedPosteriorsAndLikelihood) {
    const LogHmm h = toy();
    EXPECT_NEAR(logLikelihood(h), std::log(0.199), 1e-12);
    auto g = posteriors(h);
    ASSERT_EQ(g.size(), 2u);
    EXPECT_NEAR(g[0][0], (0.063 + 0.108) / 0.199, 1e-12);
    EXPECT_NEAR(g[0][1], (0.004 + 0.024) / 0.199, 1e-12);
    EXPECT_NEAR(g[1][1], (0.108 + 0.024) / 0.199, 1e-12);
    EXPECT_NEAR(g[1][0] + g[1][1], 1.0, 1e-12);
}

TEST(Viterbi, TransitionsOverrideLocalEmissionPreference) {
    // Step 0 has two states, step 1 has one (state sets vary per mention).
    //   b_0 = [0.6, 0.4], b_1 = [1.0], A = [[0.2], [0.9]]
    //   path (0) = .5*.6*.2 = 0.06, path (1) = .5*.4*.9 = 0.18
    // Greedy decoding would pick state 0; the coherent path picks state 1.
    LogHmm h;
    h.logEmit = {logv({0.6, 0.4}), logv({1.0})};
    h.logTrans = {logm({{0.2}, {0.9}})};
    auto r = viterbi(h);
    EXPECT_EQ(r.path, (std::vector<std::size_t>{1, 0}));
    EXPECT_NEAR(r.logScore, std::log(0.18), 1e-12);
    EXPECT_NEAR(posteriors(h)[0][1], 0.18 / 0.24, 1e-12);
}

TEST(Viterbi, ThreeStepsWithExplicitInitialMatchesBruteForce) {
    LogHmm h;
    h.logInitial = logv({0.3, 0.7});
    h.logEmit = {logv({0.5, 0.5}), logv({0.1, 0.6, 0.3}), logv({0.7, 0.3})};
    h.logTrans = {logm({{0.1, 0.2, 0.7}, {0.5, 0.4, 0.1}}), logm({{0.9, 0.1}, {0.2, 0.8}, {0.5, 0.5}})};

    double best = -1e300, total = 0.0;
    std::vector<std::size_t> bestPath;
    for (std::size_t a = 0; a < 2; ++a)
        for (std::size_t b = 0; b < 3; ++b)
            for (std::size_t c = 0; c < 2; ++c) {
                double s = h.logInitial[a] + h.logEmit[0][a] + h.logTrans[0][a][b] + h.logEmit[1][b] +
                           h.logTrans[1][b][c] + h.logEmit[2][c];
                total += std::exp(s);
                if (s > best) {
                    best = s;
                    bestPath = {a, b, c};
                }
            }
    auto r = viterbi(h);
    EXPECT_EQ(r.path, bestPath);
    EXPECT_NEAR(r.logScore, best, 1e-12);
    EXPECT_NEAR(logLikelihood(h), std::log(total), 1e-12);
}

TEST(Viterbi, EdgeCasesAndValidation) {
    LogHmm empty;
    EXPECT_TRUE(viterbi(empty).path.empty());

    LogHmm single;
    single.logEmit = {logv({0.2, 0.5, 0.3})};
    EXPECT_EQ(viterbi(single).path, std::vector<std::size_t>{1});

    LogHmm bad = toy();
    bad.logTrans.clear();
    EXPECT_THROW(viterbi(bad), std::invalid_argument);
    bad = toy();
    bad.logTrans[0][0].pop_back();
    EXPECT_THROW(viterbi(bad), std::invalid_argument);
}

TEST(LogSumExp, StableForLargeMagnitudes) {
    EXPECT_NEAR(logSumExp({1000.0, 1000.0}), 1000.0 + std::log(2.0), 1e-9);
    EXPECT_NEAR(logSumExp({-1000.0, -1000.0}), -1000.0 + std::log(2.0), 1e-9);
    EXPECT_TRUE(std::isinf(logSumExp({})));
}

TEST(Transitions, CoherenceFavoursNearbyAndSameCountry) {
    Entry naplesIt{"naples-it", "Naples", {}, "IT", "CAM", 40.85, 14.27, 910000, 'P'};
    Entry romeIt{"rome-it", "Rome", {}, "IT", "LAZ", 41.90, 12.50, 2800000, 'P'};
    Entry romeGa{"rome-us-ga", "Rome", {}, "US", "GA", 34.26, -85.16, 37000, 'P'};
    Entry naplesFl{"naples-us-fl", "Naples", {}, "US", "FL", 26.14, -81.79, 22000, 'P'};
    Entry georgia{"us-ga", "Georgia", {}, "US", "GA", 32.70, -83.40, 11000000, 'A'};
    Gazetteer g({naplesIt, romeIt, romeGa, naplesFl, georgia});

    TransitionParams p;
    EXPECT_GT(coherence(naplesIt, romeIt, p), coherence(naplesIt, romeGa, p));
    EXPECT_GT(coherence(naplesFl, romeGa, p), coherence(naplesFl, romeIt, p));
    // Containment bonus: Georgia (state) and Rome, GA.
    const double dRome = haversineKm(georgia.lat, georgia.lon, romeGa.lat, romeGa.lon);
    EXPECT_NEAR(coherence(georgia, romeGa, p),
                -p.lambda * std::log1p(dRome / p.scaleKm) + p.sameCountry + p.containment, 1e-12);
    const double dNaples = haversineKm(georgia.lat, georgia.lon, naplesFl.lat, naplesFl.lon);
    EXPECT_NEAR(coherence(georgia, naplesFl, p), -p.lambda * std::log1p(dNaples / p.scaleKm) + p.sameCountry, 1e-12);

    // Gazetteer normalization: every row of the full transition matrix sums to 1.
    TransitionModel full(g, p);
    std::vector<int> all = {0, 1, 2, 3, 4};
    auto m = full.logMatrix(g, all, all);
    for (const auto& row : m) EXPECT_NEAR(std::exp(logSumExp(row)), 1.0, 1e-12);

    // Local normalization renormalizes over the candidate subset only.
    p.normalization = Normalization::Local;
    TransitionModel local(g, p);
    auto sub = local.logMatrix(g, {0}, {1, 2});
    EXPECT_NEAR(std::exp(sub[0][0]) + std::exp(sub[0][1]), 1.0, 1e-12);
    EXPECT_GT(sub[0][0], sub[0][1]);
}
