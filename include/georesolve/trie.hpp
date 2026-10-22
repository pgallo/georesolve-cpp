#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace georesolve {

// Token-level trie: keys are sequences of normalized tokens, values are gazetteer
// indices. Matching on tokens (not characters) means "New York City" and
// "New York" share a path and the longest match wins naturally.
class TokenTrie {
public:
    struct Match {
        std::size_t length = 0;                 // tokens consumed, 0 when no match
        const std::vector<int>* values = nullptr;
    };

    TokenTrie() : nodes_(1) {}

    // Inserting the same (key, value) twice is a no-op.
    void insert(const std::vector<std::string>& key, int value);

    // Longest key that is a prefix of keys[start..].
    Match longestMatch(const std::vector<std::string>& keys, std::size_t start) const;

    std::size_t nodeCount() const { return nodes_.size(); }

private:
    struct Node {
        std::unordered_map<std::string, std::size_t> next;
        std::vector<int> values;
    };
    std::vector<Node> nodes_;
};

}  // namespace georesolve
