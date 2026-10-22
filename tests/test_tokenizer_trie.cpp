#include <gtest/gtest.h>

#include "georesolve/gazetteer.hpp"
#include "georesolve/json.hpp"
#include "georesolve/matcher.hpp"
#include "georesolve/tokenizer.hpp"
#include "georesolve/trie.hpp"

using namespace georesolve;

namespace {

Entry place(const std::string& id, const std::string& name, const std::string& cc, const std::string& admin1,
            std::vector<std::string> alts = {}, char fc = 'P') {
    Entry e;
    e.id = id;
    e.name = name;
    e.altNames = std::move(alts);
    e.country = cc;
    e.admin1 = admin1;
    e.population = 1000;
    e.featureClass = fc;
    return e;
}

}  // namespace

TEST(Tokenizer, OffsetsCapitalizationAndPunctuation) {
    auto t = tokenize("From Paris, Texas to St. Louis.");
    ASSERT_EQ(t.size(), 6u);
    EXPECT_EQ(t[1].text, "Paris");
    EXPECT_EQ(t[1].begin, 5u);
    EXPECT_EQ(t[1].end, 10u);
    EXPECT_TRUE(t[1].capitalized);
    EXPECT_FALSE(t[3].capitalized);  // "to"
    EXPECT_EQ(t[2].punctBefore, ',');
    EXPECT_EQ(t[5].norm, "louis");
    EXPECT_EQ(t[5].punctBefore, '.');
}

TEST(Tokenizer, FoldsLatin1Diacritics) {
    EXPECT_EQ(normalize("Córdoba"), "cordoba");
    EXPECT_EQ(normalize("MÉRIDA"), "merida");
    EXPECT_EQ(normalize("Zürich"), "zurich");
    auto t = tokenize("Él vive en Málaga");
    ASSERT_EQ(t.size(), 4u);
    EXPECT_TRUE(t[0].capitalized);
    EXPECT_EQ(t[3].norm, "malaga");
}

TEST(Tokenizer, ContextTermsDropStopwordsAndMention) {
    auto t = tokenize("The pizza in Naples is baked in 900 degree ovens");
    auto terms = contextTerms(t, 3, 4);
    std::vector<std::string> want = {"pizza", "baked", "degree", "ovens"};
    EXPECT_EQ(terms, want);
}

TEST(TokenTrie, LongestMatchWins) {
    TokenTrie trie;
    trie.insert({"new", "york"}, 1);
    trie.insert({"new", "york", "city"}, 2);
    trie.insert({"york"}, 3);

    std::vector<std::string> keys = {"in", "new", "york", "city", "today"};
    auto m = trie.longestMatch(keys, 1);
    EXPECT_EQ(m.length, 3u);
    ASSERT_NE(m.values, nullptr);
    EXPECT_EQ(*m.values, std::vector<int>{2});

    std::vector<std::string> shorter = {"new", "york", "state"};
    m = trie.longestMatch(shorter, 0);
    EXPECT_EQ(m.length, 2u);
    EXPECT_EQ(*m.values, std::vector<int>{1});

    EXPECT_EQ(trie.longestMatch(keys, 0).length, 0u);   // "in" starts nothing
    std::vector<std::string> partial = {"new", "jersey"};
    EXPECT_EQ(trie.longestMatch(partial, 0).length, 0u);  // a prefix node without a value is not a match
}

TEST(TokenTrie, SharedKeyAccumulatesValuesWithoutDuplicates) {
    TokenTrie trie;
    trie.insert({"springfield"}, 10);
    trie.insert({"springfield"}, 11);
    trie.insert({"springfield"}, 10);
    std::vector<std::string> keys = {"springfield"};
    auto m = trie.longestMatch(keys, 0);
    ASSERT_NE(m.values, nullptr);
    EXPECT_EQ(*m.values, (std::vector<int>{10, 11}));
}

TEST(MentionMatcher, MultiWordAlternateNamesAndCapitalizationFilter) {
    Gazetteer g({
        place("nyc", "New York", "US", "NY", {"New York City", "NYC"}),
        place("ny", "New York", "US", "NY", {}, 'A'),
        place("stl", "St. Louis", "US", "MO", {"Saint Louis"}),
        place("nice", "Nice", "FR", "PAC"),
        place("cba", "Córdoba", "ES", "AN"),
    });
    MentionMatcher matcher(g);

    std::string text = "A nice trip from New York City to Saint Louis, then St. Louis and Cordoba. Nice!";
    auto mentions = matcher.find(text, tokenize(text));
    ASSERT_EQ(mentions.size(), 5u);

    EXPECT_EQ(mentions[0].surface, "New York City");
    EXPECT_EQ(mentions[0].candidates, std::vector<int>{0});
    EXPECT_EQ(mentions[1].surface, "Saint Louis");
    EXPECT_EQ(mentions[2].surface, "St. Louis");
    EXPECT_EQ(mentions[2].candidates, std::vector<int>{2});
    EXPECT_EQ(mentions[3].surface, "Cordoba");  // accent-insensitive
    EXPECT_EQ(mentions[4].surface, "Nice");     // lower-case "nice" earlier was skipped

    std::string ambiguous = "Back in New York.";
    auto m2 = matcher.find(ambiguous, tokenize(ambiguous));
    ASSERT_EQ(m2.size(), 1u);
    EXPECT_EQ(m2[0].candidates, (std::vector<int>{0, 1}));
    EXPECT_EQ(ambiguous.substr(m2[0].charBegin, m2[0].charEnd - m2[0].charBegin), "New York");
}

TEST(Gazetteer, ContainmentAndLookup) {
    Gazetteer g({
        place("us", "United States", "US", "", {}, 'A'),
        place("us-tx", "Texas", "US", "TX", {}, 'A'),
        place("paris-tx", "Paris", "US", "TX"),
        place("paris-fr", "Paris", "FR", "IDF"),
    });
    EXPECT_EQ(g.indexOf("us-tx"), 1);
    EXPECT_EQ(g.indexOf("nope"), -1);
    EXPECT_TRUE(Gazetteer::contains(g.at(0), g.at(2)));
    EXPECT_TRUE(Gazetteer::contains(g.at(0), g.at(1)));
    EXPECT_TRUE(Gazetteer::contains(g.at(1), g.at(2)));
    EXPECT_FALSE(Gazetteer::contains(g.at(1), g.at(3)));
    EXPECT_FALSE(Gazetteer::contains(g.at(2), g.at(1)));  // a city contains nothing
    EXPECT_FALSE(Gazetteer::contains(g.at(1), g.at(0)));  // a state does not contain its country
    EXPECT_EQ(g.describe(2), "Paris, TX, US");
    EXPECT_THROW(Gazetteer({place("x", "X", "US", ""), place("x", "Y", "US", "")}), std::runtime_error);
}

TEST(Gazetteer, BundledCsvLoadsAndIsAmbiguous) {
    auto g = Gazetteer::loadCsv(std::string(GEORESOLVE_DATA_DIR) + "/gazetteer.csv");
    EXPECT_GE(g.size(), 300u);
    EXPECT_LE(g.size(), 600u);
    MentionMatcher matcher(g);
    std::string text = "Springfield";
    auto m = matcher.find(text, tokenize(text));
    ASSERT_EQ(m.size(), 1u);
    EXPECT_GE(m[0].candidates.size(), 5u);
}

TEST(Json, ParsesCorpusLine) {
    auto v = json::parse(R"({"text": "Café in \"Paris\"", "gold": [["Paris", "paris-fr"]], "n": -1.5e1})");
    EXPECT_EQ(v.at("text").string, "Café in \"Paris\"");
    ASSERT_TRUE(v.at("gold").isArray());
    EXPECT_EQ(v.at("gold").array[0].array[1].string, "paris-fr");
    EXPECT_DOUBLE_EQ(v.at("n").number, -15.0);
    EXPECT_THROW(json::parse("{\"a\": 1,}"), json::ParseError);
    EXPECT_THROW(json::parse("[1, 2"), json::ParseError);
}
