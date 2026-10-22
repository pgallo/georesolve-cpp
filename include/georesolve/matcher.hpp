#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "georesolve/gazetteer.hpp"
#include "georesolve/tokenizer.hpp"
#include "georesolve/trie.hpp"

namespace georesolve {

struct Mention {
    std::size_t tokenBegin = 0;  // [tokenBegin, tokenEnd) into the token vector
    std::size_t tokenEnd = 0;
    std::size_t charBegin = 0;   // byte span in the input text
    std::size_t charEnd = 0;
    std::string surface;
    std::vector<int> candidates;  // gazetteer indices, never empty
};

// Finds gazetteer names (primary and alternate) in tokenized text.
// A mention must start with a capitalized token, which filters most common-noun
// collisions without a part-of-speech tagger.
class MentionMatcher {
public:
    explicit MentionMatcher(const Gazetteer& gazetteer);

    std::vector<Mention> find(std::string_view text, const std::vector<Token>& tokens) const;

    const TokenTrie& trie() const { return trie_; }

private:
    TokenTrie trie_;
};

}  // namespace georesolve
