#include "georesolve/gazetteer.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace georesolve {

namespace {

// Splits one CSV record. Supports double-quoted fields with "" escapes.
std::vector<std::string> splitCsv(const std::string& line) {
    std::vector<std::string> fields;
    std::string cur;
    bool quoted = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (quoted) {
            if (c == '"' && i + 1 < line.size() && line[i + 1] == '"') {
                cur += '"';
                ++i;
            } else if (c == '"') {
                quoted = false;
            } else {
                cur += c;
            }
        } else if (c == '"') {
            quoted = true;
        } else if (c == ',') {
            fields.push_back(std::move(cur));
            cur.clear();
        } else if (c != '\r') {
            cur += c;
        }
    }
    fields.push_back(std::move(cur));
    return fields;
}

std::vector<std::string> splitAlt(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string part;
    while (std::getline(ss, part, '|')) {
        if (!part.empty()) out.push_back(part);
    }
    return out;
}

}  // namespace

Gazetteer::Gazetteer(std::vector<Entry> entries) : entries_(std::move(entries)) {
    for (std::size_t i = 0; i < entries_.size(); ++i) {
        auto [it, inserted] = byId_.emplace(entries_[i].id, static_cast<int>(i));
        if (!inserted) throw std::runtime_error("duplicate gazetteer id: " + entries_[i].id);
    }
}

Gazetteer Gazetteer::loadCsv(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open gazetteer: " + path);
    std::string line;
    std::getline(in, line);  // header
    std::vector<Entry> entries;
    std::size_t lineNo = 1;
    while (std::getline(in, line)) {
        ++lineNo;
        if (line.empty()) continue;
        auto f = splitCsv(line);
        if (f.size() != 9) {
            throw std::runtime_error("gazetteer line " + std::to_string(lineNo) + ": expected 9 fields");
        }
        Entry e;
        e.id = f[0];
        e.name = f[1];
        e.altNames = splitAlt(f[2]);
        e.country = f[3];
        e.admin1 = f[4];
        e.lat = std::stod(f[5]);
        e.lon = std::stod(f[6]);
        e.population = std::stoll(f[7]);
        e.featureClass = f[8].empty() ? 'P' : f[8][0];
        entries.push_back(std::move(e));
    }
    return Gazetteer(std::move(entries));
}

int Gazetteer::indexOf(const std::string& id) const {
    auto it = byId_.find(id);
    return it == byId_.end() ? -1 : it->second;
}

std::string Gazetteer::describe(int index) const {
    const Entry& e = at(index);
    std::string s = e.name;
    if (!e.admin1.empty() && e.featureClass == 'P') s += ", " + e.admin1;
    s += ", " + e.country;
    return s;
}

bool Gazetteer::contains(const Entry& area, const Entry& place) {
    if (!area.isAdminArea() || area.id == place.id || area.country != place.country) return false;
    if (area.admin1.empty()) return true;  // a country contains everything with its code
    return place.admin1 == area.admin1 && !place.isCountry();
}

}  // namespace georesolve
