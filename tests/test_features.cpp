#include <gtest/gtest.h>

#include <cmath>

#include "georesolve/features.hpp"
#include "georesolve/matcher.hpp"

using namespace georesolve;

namespace {

Gazetteer smallGazetteer() {
    return Gazetteer({
        {"paris-fr", "Paris", {}, "FR", "IDF", 48.86, 2.35, 2100000, 'P'},
        {"paris-us-tx", "Paris", {}, "US", "TX", 33.66, -95.56, 25000, 'P'},
        {"us-tx", "Texas", {}, "US", "TX", 31.50, -99.30, 30500000, 'A'},
        {"dallas-us-tx", "Dallas", {}, "US", "TX", 32.78, -96.80, 1300000, 'P'},
    });
}

struct Fixture {
    Gazetteer g = smallGazetteer();
    MentionMatcher matcher{g};
    std::vector<Token> tokens;
    std::vector<Mention> mentions;
    std::vector<std::vector<FeatureVector>> features;

    explicit Fixture(const std::string& text) {
        tokens = tokenize(text);
        mentions = matcher.find(text, tokens);
        std::vector<std::vector<double>> knn;
        for (const auto& m : mentions) knn.emplace_back(m.candidates.size(), 1.0 / static_cast<double>(m.candidates.size()));
        features = extractFeatures(g, tokens, mentions, knn);
    }
};

}  // namespace

TEST(Features, PopulationPriorIsRelativeToBestCandidate) {
    Fixture f("Paris is busy.");
    ASSERT_EQ(f.mentions.size(), 1u);
    const auto& c = f.features[0];
    EXPECT_DOUBLE_EQ(c[0][kLogPopulation], 0.0);  // Paris, FR is the most populous
    EXPECT_NEAR(c[1][kLogPopulation], std::log10(25001.0) - std::log10(2100001.0), 1e-12);
    EXPECT_NEAR(c[0][kLogKnn], std::log(0.5), 1e-12);
}

TEST(Features, CommaQualifierLinksCityAndState) {
    Fixture f("Paris, Texas has a rodeo.");
    ASSERT_EQ(f.mentions.size(), 2u);
    EXPECT_DOUBLE_EQ(f.features[0][0][kQualifiedBy], 0.0);  // Paris, FR
    EXPECT_DOUBLE_EQ(f.features[0][1][kQualifiedBy], 1.0);  // Paris, TX
    EXPECT_DOUBLE_EQ(f.features[1][0][kQualifies], 1.0);    // Texas qualifies a Paris candidate
    // Texas is also an unambiguous anchor containing Paris, TX.
    EXPECT_DOUBLE_EQ(f.features[0][1][kAnchorContains], 1.0);
    EXPECT_DOUBLE_EQ(f.features[0][0][kAnchorContains], 0.0);
}

TEST(Features, QualifierRequiresAdjacency) {
    Fixture f("Paris is far from Texas.");
    ASSERT_EQ(f.mentions.size(), 2u);
    EXPECT_DOUBLE_EQ(f.features[0][1][kQualifiedBy], 0.0);
    EXPECT_DOUBLE_EQ(f.features[0][1][kAnchorContains], 1.0);  // document-level evidence still applies
}

TEST(Features, AnchorProximityAndCountry) {
    Fixture f("From Dallas we drove to Paris.");
    ASSERT_EQ(f.mentions.size(), 2u);
    const auto& paris = f.features[1];
    EXPECT_DOUBLE_EQ(paris[1][kAnchorSameCountry], 1.0);
    EXPECT_DOUBLE_EQ(paris[0][kAnchorSameCountry], 0.0);
    EXPECT_GT(paris[1][kAnchorProximity], paris[0][kAnchorProximity]);
    EXPECT_LT(paris[0][kAnchorProximity], -1.5);  // thousands of km away
}

TEST(EmissionModel, LearnsToPreferInformativeFeature) {
    // Two candidates; the gold one always has kAnchorContains = 1 but lower population.
    std::vector<EmissionModel::Instance> data;
    for (int i = 0; i < 20; ++i) {
        FeatureVector big{}, small{};
        big[kLogPopulation] = 0.0;
        small[kLogPopulation] = -1.5;
        small[kAnchorContains] = 1.0;
        data.push_back({{big, small}, 1});
    }
    // And a few where population alone decides.
    for (int i = 0; i < 20; ++i) {
        FeatureVector big{}, small{};
        small[kLogPopulation] = -1.5;
        data.push_back({{big, small}, 0});
    }
    EmissionModel model;
    const double ll = model.fit(data, 2000, 0.5, 1e-4);
    EXPECT_GT(ll, std::log(0.8));
    EXPECT_GT(model.weights()[kAnchorContains], 1.5);
    EXPECT_GT(model.weights()[kLogPopulation], 0.0);

    auto lp = model.logProbs(data[0].candidates);
    EXPECT_GT(lp[1], lp[0]);
    EXPECT_NEAR(std::exp(lp[0]) + std::exp(lp[1]), 1.0, 1e-12);
}
