#include <iostream>

#include "kag_agent/market.hpp"

namespace {
int failures = 0;
void check(bool ok, const char* message) {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL: " << message << "\n";
    }
}
}

int main() {
    using namespace kag;
    const int base[kProducts] = {25,35,60,120,250,50,160,200,100};
    for (int p = 0; p < kProducts; ++p) {
        const auto product = static_cast<Product>(p);
        check(market_price(product, 10000) == base[p], "equilibrium base price");
        check(market_price(product, 9999) >= base[p], "scarcity price monotonicity");
        check(market_price(product, 10001) <= base[p], "glut price monotonicity");
        check(market_price(product, 1000000) >= 1, "price floor");
    }
    check(market_price(MELON, 10300) == 1, "melon configured 300-unit crash reaches floor");
    check(market_price(WOOL, 10105) == 1, "wool configured 105-unit crash reaches floor");
    if (failures) return 1;
    std::cout << "market parity smoke OK\n";
    return 0;
}
