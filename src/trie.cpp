#include "georesolve/trie.hpp"

#include <algorithm>

namespace georesolve {

void TokenTrie::insert(const std::vector<std::string>& key, int value) {
    if (key.empty()) return;
    std::size_t node = 0;
    for (const auto& tok : key) {
        auto it = nodes_[node].next.find(tok);
        if (it == nodes_[node].next.end()) {
            nodes_.emplace_back();
            std::size_t child = nodes_.size() - 1;
            nodes_[node].next.emplace(tok, child);
            node = child;
        } else {
            node = it->second;
        }
    }
    auto& vals = nodes_[node].values;
    if (std::find(vals.begin(), vals.end(), value) == vals.end()) vals.push_back(value);
}

TokenTrie::Match TokenTrie::longestMatch(const std::vector<std::string>& keys, std::size_t start) const {
    Match best;
    std::size_t node = 0;
    for (std::size_t i = start; i < keys.size(); ++i) {
        auto it = nodes_[node].next.find(keys[i]);
        if (it == nodes_[node].next.end()) break;
        node = it->second;
        if (!nodes_[node].values.empty()) {
            best.length = i - start + 1;
            best.values = &nodes_[node].values;
        }
    }
    return best;
}

}  // namespace georesolve
