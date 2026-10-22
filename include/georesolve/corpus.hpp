#pragma once

#include <string>
#include <vector>

namespace georesolve {

struct GoldMention {
    std::string surface;
    std::string id;
};

struct Document {
    std::string text;
    std::vector<GoldMention> gold;
};

// One JSON object per line: {"text": "...", "gold": [["Naples", "naples-us-fl"], ...]}
// Gold pairs are listed in order of appearance. Blank lines are skipped.
std::vector<Document> loadJsonl(const std::string& path);

}  // namespace georesolve
