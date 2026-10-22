#include "georesolve/geo.hpp"

#include <algorithm>
#include <cmath>

namespace georesolve {

double haversineKm(double lat1, double lon1, double lat2, double lon2) {
    constexpr double kDegToRad = 3.14159265358979323846 / 180.0;
    const double phi1 = lat1 * kDegToRad;
    const double phi2 = lat2 * kDegToRad;
    const double dPhi = (lat2 - lat1) * kDegToRad;
    const double dLambda = (lon2 - lon1) * kDegToRad;
    const double a = std::sin(dPhi / 2) * std::sin(dPhi / 2) +
                     std::cos(phi1) * std::cos(phi2) * std::sin(dLambda / 2) * std::sin(dLambda / 2);
    // Clamp guards against a > 1 from rounding for antipodal points.
    return 2.0 * kEarthRadiusKm * std::asin(std::sqrt(std::clamp(a, 0.0, 1.0)));
}

}  // namespace georesolve
