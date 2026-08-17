#include "../include/bonsai_kg.hpp"
#include <iostream>

int main(){
    using namespace bonsaikg;
    std::cout << "Header=" << sizeof(Header) << "\n";
    std::cout << "Node=" << sizeof(Node) << "\n";
    std::cout << "BbDecl=" << sizeof(BbDecl) << "\n";
    std::cout << "Predicate=" << sizeof(Predicate) << "\n";
    std::cout << "Lifecycle=" << sizeof(Lifecycle) << "\n";
    std::cout << "Effect=" << sizeof(Effect) << "\n";
    std::cout << "NodeState=" << sizeof(NodeState) << "\n";
    std::cout << "UnitClaim=" << sizeof(UnitClaim) << "\n";
    std::cout << "MarketClaim=" << sizeof(MarketClaim) << "\n";
    std::cout << "RuntimeState=" << sizeof(RuntimeState) << "\n";
    std::cout << "ActionResult=" << sizeof(ActionResult) << "\n";
    std::cout << "EvalFrame=" << sizeof(EvalFrame) << "\n";
    std::cout << "HaltFrame=" << sizeof(HaltFrame) << "\n";
    std::cout << "max_blackboard_bytes=" << MAX_BB * sizeof(int32_t) << "\n";
    std::cout << "max_node_state_bytes=" << MAX_STATE * sizeof(NodeState) << "\n";
    std::cout << "intent_claim_bytes=" << MAX_UNITS * sizeof(UnitClaim) + MAX_MARKET_INTENTS * sizeof(MarketClaim) << "\n";
}
