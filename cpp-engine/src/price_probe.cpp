#include "sim.hpp"
#include <iostream>

int main() {
    int item = 0;
    int inventory = 0;
    while (std::cin >> item >> inventory) {
        if (item < 0 || item >= kag::N_PRODUCTS) return 2;
        std::cout << kag::market_price(item, inventory) << '\n';
    }
    return 0;
}
