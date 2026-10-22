#pragma once

namespace georesolve {

constexpr double kEarthRadiusKm = 6371.0088;  // mean Earth radius (IUGG)

// Great-circle distance in kilometres between two WGS84 points given in degrees.
double haversineKm(double lat1, double lon1, double lat2, double lon2);

}  // namespace georesolve
