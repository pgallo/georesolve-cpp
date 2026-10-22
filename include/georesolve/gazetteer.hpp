#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace georesolve {

struct Entry {
    std::string id;
    std::string name;
    std::vector<std::string> altNames;
    std::string country;  // ISO 3166-1 alpha-2
    std::string admin1;   // first-level subdivision code, empty for countries
    double lat = 0.0;
    double lon = 0.0;
    long long population = 0;
    char featureClass = 'P';  // 'A' administrative area / country, 'P' populated place

    bool isAdminArea() const { return featureClass == 'A'; }
    bool isCountry() const { return featureClass == 'A' && admin1.empty(); }
};

class Gazetteer {
public:
    Gazetteer() = default;
    explicit Gazetteer(std::vector<Entry> entries);

    // CSV columns: id,name,alt_names,country,admin1,lat,lon,population,feature_class
    // alt_names is '|'-separated. Throws std::runtime_error on malformed input.
    static Gazetteer loadCsv(const std::string& path);

    const std::vector<Entry>& entries() const { return entries_; }
    const Entry& at(int index) const { return entries_.at(static_cast<std::size_t>(index)); }
    std::size_t size() const { return entries_.size(); }

    // -1 when the id is unknown.
    int indexOf(const std::string& id) const;

    // "Naples, CAM, IT" style label.
    std::string describe(int index) const;

    // True when `area` is an administrative area (country or admin1) that contains `place`.
    static bool contains(const Entry& area, const Entry& place);

private:
    std::vector<Entry> entries_;
    std::unordered_map<std::string, int> byId_;
};

}  // namespace georesolve
