#include "georesolve/matcher.hpp"

namespace georesolve {

MentionMatcher::MentionMatcher(const Gazetteer& gazetteer) {
    for (std::size_t i = 0; i < gazetteer.size(); ++i) {
        const Entry& e = gazetteer.entries()[i];
        trie_.insert(normalizedTokens(e.name), static_cast<int>(i));
        for (const auto& alt : e.altNames) trie_.insert(normalizedTokens(alt), static_cast<int>(i));
    }
}

std::vector<Mention> MentionMatcher::find(std::string_view text, const std::vector<Token>& tokens) const {
    std::vector<std::string> norms;
    norms.reserve(tokens.size());
    for (const auto& t : tokens) norms.push_back(t.norm);

    std::vector<Mention> mentions;
    std::size_t i = 0;
    while (i < tokens.size()) {
        if (!tokens[i].capitalized) {
            ++i;
            continue;
        }
        auto match = trie_.longestMatch(norms, i);
        if (match.length == 0) {
            ++i;
            continue;
        }
        Mention m;
        m.tokenBegin = i;
        m.tokenEnd = i + match.length;
        m.charBegin = tokens[i].begin;
        m.charEnd = tokens[m.tokenEnd - 1].end;
        m.surface = std::string(text.substr(m.charBegin, m.charEnd - m.charBegin));
        m.candidates = *match.values;
        mentions.push_back(std::move(m));
        i += match.length;
    }
    return mentions;
}

}  // namespace georesolve
