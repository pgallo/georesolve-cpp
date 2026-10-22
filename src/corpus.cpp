#include "georesolve/corpus.hpp"

#include <fstream>
#include <stdexcept>

#include "georesolve/json.hpp"

namespace georesolve {

std::vector<Document> loadJsonl(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open corpus: " + path);
    std::vector<Document> docs;
    std::string line;
    std::size_t lineNo = 0;
    while (std::getline(in, line)) {
        ++lineNo;
        if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
        try {
            json::Value v = json::parse(line);
            Document d;
            d.text = v.at("text").string;
            for (const auto& pair : v.at("gold").array) {
                if (!pair.isArray() || pair.array.size() != 2) throw json::ParseError("gold entries are [surface, id]");
                d.gold.push_back({pair.array[0].string, pair.array[1].string});
            }
            docs.push_back(std::move(d));
        } catch (const json::ParseError& e) {
            throw std::runtime_error(path + ":" + std::to_string(lineNo) + ": " + e.what());
        }
    }
    return docs;
}

}  // namespace georesolve
