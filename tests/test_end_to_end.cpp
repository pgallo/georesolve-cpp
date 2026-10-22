#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "georesolve/corpus.hpp"
#include "georesolve/evaluation.hpp"
#include "georesolve/resolver.hpp"

using namespace georesolve;

namespace {

const std::string kData = GEORESOLVE_DATA_DIR;

// Training takes a fraction of a second; share one trained resolver across tests.
class EndToEnd : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        resolver_ = std::make_unique<Resolver>(Gazetteer::loadCsv(kData + "/gazetteer.csv"));
        report_ = resolver_->train(loadJsonl(kData + "/train.jsonl"));
    }
    static void TearDownTestSuite() { resolver_.reset(); }

    static std::string idOf(const Resolution& r) { return resolver_->gazetteer().at(r.entry).id; }

    static std::unique_ptr<Resolver> resolver_;
    static TrainingReport report_;
};

std::unique_ptr<Resolver> EndToEnd::resolver_;
TrainingReport EndToEnd::report_;

}  // namespace

TEST_F(EndToEnd, CorporaAreWellFormed) {
    auto train = loadJsonl(kData + "/train.jsonl");
    auto test = loadJsonl(kData + "/test.jsonl");
    EXPECT_GE(train.size(), 150u);
    EXPECT_GE(test.size(), 60u);
    for (const auto* docs : {&train, &test}) {
        for (const auto& d : *docs) {
            for (const auto& gm : d.gold) {
                EXPECT_GE(resolver_->gazetteer().indexOf(gm.id), 0) << gm.id;
                EXPECT_NE(d.text.find(gm.surface), std::string::npos) << d.text;
            }
        }
    }
    EXPECT_EQ(report_.alignedMentions, report_.goldMentions);
}

TEST_F(EndToEnd, ReadmeExampleSentence) {
    auto res = resolver_->resolve("I flew from Naples to Rome, then drove to Florence.");
    ASSERT_EQ(res.size(), 3u);
    EXPECT_EQ(idOf(res[0]), "naples-it");
    EXPECT_EQ(idOf(res[1]), "rome-it");
    EXPECT_EQ(idOf(res[2]), "florence-it");
    for (const auto& r : res) {
        EXPECT_GT(r.confidence, 0.5);
        EXPECT_LE(r.confidence, 1.0 + 1e-9);
        EXPECT_EQ(r.ranked.front().first, r.entry);
    }
}

TEST_F(EndToEnd, ExplicitQualifierBeatsPopulation) {
    auto res = resolver_->resolve("We rented a condo in Naples, Florida for the winter.");
    ASSERT_GE(res.size(), 2u);
    EXPECT_EQ(idOf(res[0]), "naples-us-fl");
    // The population baseline gets this one wrong.
    auto pop = resolver_->resolve("We rented a condo in Naples, Florida for the winter.", Method::Population);
    EXPECT_EQ(idOf(pop[0]), "naples-it");
}

TEST_F(EndToEnd, GeographicCoherenceAcrossMentions) {
    // No context words, no qualifier: only the sequence decides.
    auto res = resolver_->resolve("The tour visited Kingston, Ottawa and Montreal.");
    ASSERT_EQ(res.size(), 3u);
    EXPECT_EQ(idOf(res[0]), "kingston-ca-on");

    res = resolver_->resolve("We drove from Portland to Salem and then Eugene.");
    ASSERT_EQ(res.size(), 3u);
    EXPECT_EQ(idOf(res[0]), "portland-us-or");
    EXPECT_EQ(idOf(res[1]), "salem-us-or");
}

TEST_F(EndToEnd, ContextWordsShiftTheDecision) {
    auto it = resolver_->resolve("The ferry to Capri leaves Naples after an espresso.");
    auto fl = resolver_->resolve("Retirees play golf on the beaches of Naples.");
    ASSERT_EQ(it.size(), 1u);
    ASSERT_EQ(fl.size(), 1u);
    EXPECT_EQ(idOf(it[0]), "naples-it");
    EXPECT_EQ(idOf(fl[0]), "naples-us-fl");
}

TEST_F(EndToEnd, NoMentionsYieldsEmptyResult) {
    EXPECT_TRUE(resolver_->resolve("nothing to see here, just lower-case words.").empty());
    EXPECT_TRUE(resolver_->resolve("").empty());
}

TEST_F(EndToEnd, HeldOutBenchmarkHmmBeatsBaselines) {
    auto test = loadJsonl(kData + "/test.jsonl");
    const Metrics pop = evaluate(*resolver_, test, Method::Population);
    const Metrics knn = evaluate(*resolver_, test, Method::Knn);
    const Metrics hmm = evaluate(*resolver_, test, Method::Hmm);
    EXPECT_EQ(hmm.missed, 0u);
    EXPECT_GT(knn.accuracy(), pop.accuracy());
    EXPECT_GT(hmm.accuracy(), knn.accuracy());
    EXPECT_GE(hmm.accuracy(), 0.90);
    EXPECT_LT(hmm.meanErrorKm(), knn.meanErrorKm());
}
