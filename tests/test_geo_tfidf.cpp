#include <gtest/gtest.h>

#include <cmath>

#include "georesolve/geo.hpp"
#include "georesolve/knn.hpp"
#include "georesolve/tfidf.hpp"

using namespace georesolve;

namespace {
constexpr double kPi = 3.14159265358979323846;
}  // namespace

TEST(Haversine, KnownDistances) {
    EXPECT_DOUBLE_EQ(haversineKm(48.86, 2.35, 48.86, 2.35), 0.0);
    // Paris to London is about 344 km.
    EXPECT_NEAR(haversineKm(48.86, 2.35, 51.51, -0.13), 344.0, 2.0);
    // Naples (Italy) to Naples (Florida), checked against the spherical law of cosines,
    // which is numerically fine at this distance.
    const double d = kPi / 180.0;
    const double cosines = kEarthRadiusKm * std::acos(std::sin(40.85 * d) * std::sin(26.14 * d) +
                                                      std::cos(40.85 * d) * std::cos(26.14 * d) *
                                                          std::cos((-81.79 - 14.27) * d));
    EXPECT_NEAR(haversineKm(40.85, 14.27, 26.14, -81.79), cosines, 1e-6);
    EXPECT_NEAR(cosines, 8617.0, 1.0);
    // Equator to pole along a meridian is a quarter of the circumference.
    EXPECT_NEAR(haversineKm(0, 0, 90, 0), kPi * kEarthRadiusKm / 2.0, 1e-6);
    // Antipodes: half the circumference and no NaN. asin() is ill-conditioned near 1,
    // so the tolerance here is looser (about a metre).
    EXPECT_NEAR(haversineKm(10, 20, -10, -160), kPi * kEarthRadiusKm, 1e-3);
}

TEST(Haversine, SymmetricAndWrapsLongitude) {
    EXPECT_DOUBLE_EQ(haversineKm(51.5, -0.1, -33.9, 151.2), haversineKm(-33.9, 151.2, 51.5, -0.1));
    // Crossing the antimeridian: 179E to 179W at the equator is 2 degrees, not 358.
    EXPECT_NEAR(haversineKm(0, 179, 0, -179), 2.0 * kPi * kEarthRadiusKm / 180.0, 1e-6);
}

TEST(SparseVectors, DotNormCosine) {
    SparseVector a = {{0, 1.0}, {2, 2.0}, {5, 3.0}};
    SparseVector b = {{1, 4.0}, {2, 1.0}, {5, -1.0}};
    EXPECT_DOUBLE_EQ(dot(a, b), 2.0 - 3.0);
    EXPECT_DOUBLE_EQ(norm(a), std::sqrt(14.0));
    EXPECT_NEAR(cosine(a, a), 1.0, 1e-12);
    EXPECT_DOUBLE_EQ(cosine(a, {}), 0.0);
    EXPECT_NEAR(cosine(a, b), -1.0 / (std::sqrt(14.0) * std::sqrt(18.0)), 1e-12);
}

TEST(TfIdf, HandComputedWeights) {
    TfIdfVectorizer v;
    v.fit({{"beach", "golf"}, {"beach", "pizza"}, {"pizza", "pizza", "ferry"}});
    // N = 3. df(beach)=2, df(golf)=1, df(pizza)=2, df(ferry)=1.
    const double idf2 = std::log(4.0 / 3.0) + 1.0;
    const double idf1 = std::log(4.0 / 2.0) + 1.0;
    EXPECT_NEAR(v.idf("beach"), idf2, 1e-12);
    EXPECT_NEAR(v.idf("golf"), idf1, 1e-12);
    EXPECT_DOUBLE_EQ(v.idf("unknown"), 0.0);

    // "pizza pizza golf": tf(pizza)=2 -> 1 + ln 2 (sublinear), tf(golf)=1.
    auto x = v.transform({"pizza", "pizza", "golf", "unknown"});
    ASSERT_EQ(x.size(), 2u);
    const double wp = (1.0 + std::log(2.0)) * idf2;
    const double wg = idf1;
    const double n = std::sqrt(wp * wp + wg * wg);
    double gotPizza = 0, gotGolf = 0;
    for (auto [id, w] : x) {
        if (std::abs(w - wp / n) < 1e-12) gotPizza = w;
        if (std::abs(w - wg / n) < 1e-12) gotGolf = w;
    }
    EXPECT_GT(gotPizza, 0.0);
    EXPECT_GT(gotGolf, 0.0);
    EXPECT_NEAR(norm(x), 1.0, 1e-12);
    EXPECT_TRUE(v.transform({"nothing", "known"}).empty());
}

TEST(KnnContextClassifier, VotesForCandidateWithSimilarContexts) {
    // Labels 1 (a coastal resort) and 2 (a historic city).
    KnnContextClassifier knn(3, 0.1);
    knn.fit({
        {0, 1, {"golf", "beach", "condo"}},
        {1, 1, {"beach", "retirees", "gulf"}},
        {2, 2, {"pizza", "ferry", "vesuvius"}},
        {3, 2, {"pizza", "espresso", "churches"}},
        {4, 7, {"golf", "beach"}},  // label not among the candidates below: must be ignored
    });
    auto p = knn.predict(knn.vectorize({"rented", "beach", "condo"}), {1, 2});
    ASSERT_EQ(p.size(), 2u);
    EXPECT_NEAR(p[0] + p[1], 1.0, 1e-12);
    EXPECT_GT(p[0], 0.8);

    p = knn.predict(knn.vectorize({"pizza", "ferry"}), {1, 2});
    EXPECT_GT(p[1], p[0]);

    // No overlapping vocabulary: uniform.
    p = knn.predict(knn.vectorize({"snow"}), {1, 2});
    EXPECT_DOUBLE_EQ(p[0], 0.5);

    // Leave-one-document-out: excluding doc 0 removes the only "condo" neighbour, but
    // doc 1 still shares "beach".
    auto loo = knn.predict(knn.vectorize({"condo", "beach"}), {1, 2}, 0);
    EXPECT_GT(loo[0], loo[1]);
}
